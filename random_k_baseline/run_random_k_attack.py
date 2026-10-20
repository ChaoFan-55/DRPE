from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torchvision.utils import save_image


# ============================================================
# Project path
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# Reuse existing project modules
# ============================================================

from src.attacks.inversion import compute_named_gradients, total_variation
from src.attacks.metrics import OptionalLPIPS, mse_01, psnr_01, ssim_01
from src.data import build_attack_dataset, get_dataset_spec
from src.models import build_model
from src.utils import ensure_dir, resolve_device, set_global_seed


# ============================================================
# Random-k
# ============================================================

def stable_seed(*parts: object) -> int:
    """
    Generate a deterministic seed.

    Random-k mask is determined by:
        random_k_seed
        round_index
        client_id
        tensor_name
    """
    text = "|".join(str(x) for x in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()

    return (
        int.from_bytes(
            digest[:8],
            byteorder="little",
            signed=False,
        )
        % (2**63 - 1)
    )


def build_masks(
    raw_grads,
    keep_ratio: float,
    base_seed: int,
    round_index: int,
    client_id: int,
    device: torch.device,
):
    """
    Pre-generate one fixed Random-k mask for every parameter tensor.

    Important:
    The mask is generated only once before the attack.
    It is NOT regenerated during every inversion iteration.
    """

    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError(
            f"keep_ratio must be in (0, 1], got {keep_ratio}"
        )

    masks = {}

    for name, grad in raw_grads.items():

        numel = grad.numel()

        selected = min(
            numel,
            max(
                1,
                int(round(numel * keep_ratio)),
            ),
        )

        seed = stable_seed(
            base_seed,
            round_index,
            client_id,
            name,
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        indices = torch.randperm(
            numel,
            generator=generator,
            device="cpu",
        )[:selected]

        masks[name] = {
            "idx": indices.to(
                device=device,
                dtype=torch.long,
            ),
            "shape": tuple(grad.shape),
            "n": numel,
            "k": selected,

            # Random-k unbiased scaling:
            #
            # p = k / n
            # 1 / p = n / k
            #
            "scale": numel / selected,
        }

    return masks


def random_k_observation(
    raw_grads,
    masks,
):
    """
    Random-k unbiased sparse observation.

    For selected coordinates:

        C(g)_i = g_i / p

    where:

        p = k / d

    Only selected values are used by the attacker.
    """

    observation = {}

    for name, grad in raw_grads.items():

        mask = masks[name]

        flat = grad.reshape(-1)

        values = flat.index_select(
            0,
            mask["idx"],
        )

        # unbiased Random-k
        if mask["k"] < mask["n"]:
            values = values * mask["scale"]

        observation[name] = values

    return observation


# ============================================================
# Matching loss
# ============================================================

def cosine_loss(
    candidate,
    target,
):
    """
    Full-model cosine gradient matching loss.

    Same idea as the cosine matching used in the
    existing DLG/iDLG experiments.
    """

    device = next(iter(candidate.values())).device

    dot = torch.zeros(
        (),
        device=device,
    )

    candidate_norm_sq = torch.zeros(
        (),
        device=device,
    )

    target_norm_sq = torch.zeros(
        (),
        device=device,
    )

    for name, current in candidate.items():

        reference = target[name].to(
            device=current.device,
            dtype=current.dtype,
        )

        dot = dot + torch.sum(
            current * reference
        )

        candidate_norm_sq = (
            candidate_norm_sq
            + torch.sum(current.square())
        )

        target_norm_sq = (
            target_norm_sq
            + torch.sum(reference.square())
        )

    denominator = (
        torch.sqrt(
            candidate_norm_sq.clamp_min(1e-18)
        )
        *
        torch.sqrt(
            target_norm_sq.clamp_min(1e-18)
        )
    )

    return (
        1.0
        -
        dot
        /
        denominator.clamp_min(1e-18)
    )


# ============================================================
# iDLG label inference for sparse Random-k observation
# ============================================================

def infer_idlg_label(
    observation,
    masks,
    num_classes: int = 10,
):
    """
    iDLG label inference adapted to sparse Random-k observations.

    First try final classifier bias.
    If the true-class bias coordinate was not selected,
    fall back to sparse classifier-weight row statistics.
    """

    # --------------------------------------------------------
    # 1. classifier bias
    # --------------------------------------------------------

    for name, mask in masks.items():

        shape = mask["shape"]

        if (
            len(shape) == 1
            and shape[0] == num_classes
            and name.endswith("bias")
        ):

            indices = (
                mask["idx"]
                .detach()
                .cpu()
            )

            values = (
                observation[name]
                .detach()
                .float()
                .cpu()
            )

            # For single-sample cross entropy,
            # true-class bias gradient should be negative.
            if torch.any(values < 0):

                local_position = int(
                    torch.argmin(values).item()
                )

                predicted_label = int(
                    indices[
                        local_position
                    ].item()
                )

                return (
                    predicted_label,
                    f"{name}:sparse_bias",
                )

    # --------------------------------------------------------
    # 2. classifier weight
    # --------------------------------------------------------

    for name, mask in masks.items():

        shape = mask["shape"]

        if (
            len(shape) == 2
            and shape[0] == num_classes
            and name.endswith("weight")
        ):

            width = shape[1]

            indices = (
                mask["idx"]
                .detach()
                .cpu()
            )

            values = (
                observation[name]
                .detach()
                .float()
                .cpu()
            )

            row_indices = torch.div(
                indices,
                width,
                rounding_mode="floor",
            )

            scores = torch.full(
                (num_classes,),
                float("inf"),
                dtype=torch.float32,
            )

            for class_id in range(num_classes):

                class_values = values[
                    row_indices == class_id
                ]

                if class_values.numel() > 0:
                    scores[class_id] = (
                        class_values.mean()
                    )

            if torch.isfinite(scores).any():

                predicted_label = int(
                    torch.argmin(scores).item()
                )

                return (
                    predicted_label,
                    f"{name}:sparse_row_mean",
                )

    raise RuntimeError(
        "iDLG could not infer label "
        "from Random-k sparse observation."
    )


# ============================================================
# CSV
# ============================================================

def append_csv(
    path: Path,
    row: dict,
):
    exists = path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys()),
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# Load model
# ============================================================

def load_model(
    checkpoint_path: str,
    device: torch.device,
):
    checkpoint = Path(checkpoint_path)

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    spec = get_dataset_spec("cifar10")

    model = build_model(
        "resnet18",
        10,
        "group",
        spec.image_size,
    ).to(device)

    state = torch.load(
        checkpoint,
        map_location=device,
    )

    # Compatibility with checkpoints that wrap state_dict
    if (
        isinstance(state, dict)
        and "model_state_dict" in state
    ):
        state = state["model_state_dict"]

    model.load_state_dict(state)

    model.eval()

    return model


# ============================================================
# Main attack
# ============================================================

def run_attack(args):

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_global_seed(
        args.seed,
        deterministic=True,
    )

    device = resolve_device(
        args.device
    )

    spec = get_dataset_spec("cifar10")
    mean = spec.mean
    std = spec.std

    output_dir = ensure_dir(
        args.output_dir
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    model = load_model(
        args.checkpoint,
        device,
    )

    # --------------------------------------------------------
    # Attack dataset
    # --------------------------------------------------------

    dataset = build_attack_dataset(
        dataset="cifar10",
        data_root=args.data_root,
        split="train",
        download=True,
    )

    image, true_label = dataset[
        args.sample_index
    ]

    target_image = (
        image
        .unsqueeze(0)
        .to(device)
    )

    target_label = torch.tensor(
        [true_label],
        dtype=torch.long,
        device=device,
    )

    # --------------------------------------------------------
    # Generate target gradient
    # --------------------------------------------------------

    target_raw_gradients = (
        compute_named_gradients(
            model=model,
            image_01=target_image,
            label=target_label,
            parameter_scope="all",
            create_graph=False,
            mean=mean,
            std=std,
        )
    )

    # --------------------------------------------------------
    # Build fixed Random-k masks
    # --------------------------------------------------------

    masks = build_masks(
        raw_grads=target_raw_gradients,
        keep_ratio=args.keep_ratio,
        base_seed=args.random_k_seed,
        round_index=args.round_index,
        client_id=args.client_id,
        device=device,
    )

    # --------------------------------------------------------
    # Simulated uploaded sparse observation
    # --------------------------------------------------------

    target_observation = {
        name: value.detach()
        for name, value
        in random_k_observation(
            target_raw_gradients,
            masks,
        ).items()
    }

    # Dense target gradients are no longer needed
    del target_raw_gradients

    # --------------------------------------------------------
    # iDLG label inference
    # --------------------------------------------------------

    inferred_label = None
    label_source = "joint_optimization"

    if args.attack == "idlg":

        inferred_label, label_source = (
            infer_idlg_label(
                target_observation,
                masks,
                num_classes=10,
            )
        )

        print(
            f"iDLG inferred label="
            f"{inferred_label}, "
            f"true={true_label}, "
            f"source={label_source}"
        )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Attack initialization
    #
    # Keep the same initialization convention as existing
    # classic attack:
    #
    # attack_seed = seed + sample_index * 1009
    # uniform initialization + sigmoid parameterization
    # --------------------------------------------------------

    attack_seed = (
        args.seed
        + args.sample_index * 1009
    )

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        attack_seed
    )

    initial_image = torch.rand(
        (1, 3, 32, 32),
        generator=generator,
        device=device,
    )

    initial_image = initial_image.clamp(
        1e-4,
        1.0 - 1e-4,
    )

    image_parameter = (
        torch.logit(initial_image)
        .detach()
        .requires_grad_(True)
    )

    # --------------------------------------------------------
    # DLG label optimization
    # --------------------------------------------------------

    label_logits = None

    optimizer_groups = [
        {
            "params": [
                image_parameter
            ],
            "lr": args.attack_lr,
        }
    ]

    if args.attack == "dlg":

        label_logits = torch.zeros(
            10,
            device=device,
            requires_grad=True,
        )

        optimizer_groups.append(
            {
                "params": [
                    label_logits
                ],
                "lr": args.label_lr,
            }
        )

    optimizer = torch.optim.Adam(
        optimizer_groups
    )

    # --------------------------------------------------------
    # Hard label for iDLG
    # --------------------------------------------------------

    hard_label_value = (
        true_label
        if inferred_label is None
        else inferred_label
    )

    hard_label = torch.tensor(
        [hard_label_value],
        dtype=torch.long,
        device=device,
    )

    # --------------------------------------------------------
    # Progress file
    # --------------------------------------------------------

    progress_path = (
        output_dir
        / f"progress_{args.attack}_random_k.csv"
    )

    if progress_path.exists():
        progress_path.unlink()

    # --------------------------------------------------------
    # Best result tracking
    # --------------------------------------------------------

    best_match = float("inf")
    best_iteration = 0
    best_label = hard_label_value

    best_image = (
        torch.sigmoid(
            image_parameter.detach()
        )
        .clone()
    )

    final_match = float("nan")

    # --------------------------------------------------------
    # Print experiment info
    # --------------------------------------------------------

    print("=" * 100)

    print(
        f"Random-k gradient inversion | "
        f"attack={args.attack.upper()} | "
        f"device={device}"
    )

    print(
        f"checkpoint={args.checkpoint}"
    )

    print(
        f"sample={args.sample_index} | "
        f"keep_ratio={args.keep_ratio} | "
        f"unbiased=True | "
        f"random_k_seed={args.random_k_seed} | "
        f"round={args.round_index} | "
        f"client={args.client_id}"
    )

    print(
        f"iterations={args.iterations} | "
        f"optimizer=Adam | "
        f"loss=cosine | "
        f"attack_lr={args.attack_lr} | "
        f"tv_weight={args.tv_weight}"
    )

    print("=" * 100)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()

    # ========================================================
    # Attack loop
    # ========================================================

    for iteration in range(
        1,
        args.iterations + 1,
    ):

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Candidate image
        # ----------------------------------------------------

        candidate_image = torch.sigmoid(
            image_parameter
        )

        # ----------------------------------------------------
        # Candidate gradient
        # ----------------------------------------------------

        candidate_raw_gradients = (
            compute_named_gradients(
                model=model,
                image_01=candidate_image,

                label=(
                    hard_label
                    if label_logits is None
                    else None
                ),

                parameter_scope="all",
                create_graph=True,
                mean=mean,
                std=std,
                label_logits=label_logits,
            )
        )

        # ----------------------------------------------------
        # Apply exactly the same fixed Random-k mask
        # ----------------------------------------------------

        candidate_observation = (
            random_k_observation(
                candidate_raw_gradients,
                masks,
            )
        )

        # ----------------------------------------------------
        # Gradient matching
        # ----------------------------------------------------

        match = cosine_loss(
            candidate_observation,
            target_observation,
        )

        tv = total_variation(
            candidate_image
        )

        objective = (
            match
            +
            args.tv_weight * tv
        )

        # ----------------------------------------------------
        # DLG label entropy regularization
        # ----------------------------------------------------

        if (
            label_logits is not None
            and args.label_entropy_weight > 0
        ):

            probabilities = torch.softmax(
                label_logits,
                dim=0,
            )

            entropy = -(
                probabilities
                *
                torch.log(
                    probabilities.clamp_min(
                        1e-12
                    )
                )
            ).sum()

            objective = (
                objective
                +
                args.label_entropy_weight
                * entropy
            )

        # ----------------------------------------------------
        # Second-order optimization
        # ----------------------------------------------------

        if label_logits is None:
            attack_parameters = [
                image_parameter
            ]
        else:
            attack_parameters = [
                image_parameter,
                label_logits,
            ]

        attack_gradients = torch.autograd.grad(
            objective,
            attack_parameters,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )

        for parameter, gradient in zip(
            attack_parameters,
            attack_gradients,
        ):
            parameter.grad = gradient

        optimizer.step()

        # ----------------------------------------------------
        # Current prediction
        # ----------------------------------------------------

        if label_logits is None:
            predicted_label = (
                hard_label_value
            )
        else:
            predicted_label = int(
                torch.argmax(
                    label_logits.detach()
                ).item()
            )

        current_match = float(
            match.detach().item()
        )

        # ----------------------------------------------------
        # Save best image according to gradient match
        # ----------------------------------------------------

        if current_match < best_match:

            best_match = current_match
            best_iteration = iteration
            best_label = predicted_label

            best_image = (
                candidate_image
                .detach()
                .clone()
            )

        final_match = current_match

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        should_print = (
            iteration == 1
            or iteration % args.print_every == 0
            or iteration == args.iterations
        )

        if should_print:

            if device.type == "cuda":
                torch.cuda.synchronize(
                    device
                )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            seconds_per_iteration = (
                elapsed / iteration
            )

            eta_seconds = (
                seconds_per_iteration
                *
                (
                    args.iterations
                    - iteration
                )
            )

            print(
                f"  attack={args.attack:4s} "
                f"obs=random_k "
                f"iter={iteration:5d}/"
                f"{args.iterations} | "
                f"match={current_match:.6e} | "
                f"label={predicted_label}/"
                f"{true_label} | "
                f"{seconds_per_iteration:.3f}s/iter | "
                f"ETA={eta_seconds / 60:.1f}m",
                flush=True,
            )

            append_csv(
                progress_path,
                {
                    "iteration": iteration,
                    "match_loss":
                        current_match,
                    "predicted_label":
                        predicted_label,
                    "best_match_loss":
                        best_match,
                    "best_iteration":
                        best_iteration,
                    "elapsed_seconds":
                        elapsed,
                },
            )

        # ----------------------------------------------------
        # Release iteration graph
        # ----------------------------------------------------

        del candidate_raw_gradients
        del candidate_observation
        del match
        del objective

    # ========================================================
    # Final result
    # ========================================================

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    attack_seconds = (
        time.perf_counter()
        - start_time
    )

    final_image = torch.sigmoid(
        image_parameter.detach()
    )

    target_cpu = (
        target_image
        .detach()
        .cpu()
    )

    best_cpu = (
        best_image
        .detach()
        .cpu()
    )

    final_cpu = (
        final_image
        .detach()
        .cpu()
    )

    # --------------------------------------------------------
    # Save images
    # --------------------------------------------------------

    save_image(
        target_cpu,
        output_dir / "target.png",
    )

    save_image(
        best_cpu,
        output_dir
        / f"reconstruction_{args.attack}_random_k_best.png",
    )

    save_image(
        final_cpu,
        output_dir
        / f"reconstruction_{args.attack}_random_k_final.png",
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    optional_lpips = OptionalLPIPS(
        device,
        enabled=not args.no_lpips,
    )

    lpips_value = optional_lpips(
        target_image,
        best_image.to(device),
    )

    result = {

        "attack_name":
            args.attack,

        "observation_mode":
            "random_k_encoded",

        "sample_index":
            args.sample_index,

        "true_label":
            int(true_label),

        "predicted_label":
            int(best_label),

        "label_correct":
            int(
                best_label
                == true_label
            ),

        "inferred_label":
            (
                None
                if inferred_label is None
                else int(inferred_label)
            ),

        "label_source":
            label_source,

        "checkpoint":
            args.checkpoint,

        "random_k_seed":
            args.random_k_seed,

        "round_index":
            args.round_index,

        "client_id":
            args.client_id,

        "keep_ratio":
            args.keep_ratio,

        "unbiased":
            True,

        "iterations":
            args.iterations,

        "best_iteration":
            best_iteration,

        "best_match_loss":
            best_match,

        "final_match_loss":
            final_match,

        "mse":
            float(
                mse_01(
                    target_cpu,
                    best_cpu,
                )
            ),

        "psnr":
            float(
                psnr_01(
                    target_cpu,
                    best_cpu,
                )
            ),

        "ssim":
            float(
                ssim_01(
                    target_cpu,
                    best_cpu,
                )
            ),

        "lpips":
            (
                None
                if lpips_value is None
                else float(lpips_value)
            ),

        "attack_seconds":
            attack_seconds,
    }

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = (
        output_dir
        / "random_k_attack_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("-" * 100)

    print(
        f"sample={args.sample_index:05d} "
        f"attack={args.attack.upper():4s} "
        f"obs=random_k | "
        f"PSNR={result['psnr']:.2f} | "
        f"SSIM={result['ssim']:.4f} | "
        f"label={best_label}/{true_label} | "
        f"best_iter={best_iteration} | "
        f"{attack_seconds:.1f}s"
    )

    print(
        f"Artifacts saved to: "
        f"{output_dir.resolve()}"
    )


# ============================================================
# CLI
# ============================================================

def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "DLG/iDLG attack against "
            "Random-k sparse gradient uploads"
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--attack",
        choices=[
            "dlg",
            "idlg",
        ],
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--random-k-seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--round-index",
        type=int,
        default=21,
    )

    parser.add_argument(
        "--client-id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--attack-lr",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--label-lr",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--tv-weight",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--label-entropy-weight",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--data-root",
        default="./data",
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--no-lpips",
        action="store_true",
    )

    return parser


if __name__ == "__main__":

    args = build_parser().parse_args()

    run_attack(args)