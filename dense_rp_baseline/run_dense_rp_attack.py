from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attacks.inversion import compute_named_gradients, total_variation
from src.attacks.metrics import OptionalLPIPS, mse_01, psnr_01, ssim_01
from src.data import build_attack_dataset, get_dataset_spec
from src.models import build_model
from src.utils import ensure_dir, resolve_device, set_global_seed

from dense_rp_baseline.dense_rp import (
    DenseRPSettings,
    clear_dense_rp_cache,
    decode_tensor,
    differentiable_encode_tensor,
    encode_tensor,
)


def append_csv(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    spec = get_dataset_spec("cifar10")
    model = build_model("resnet18", 10, "group", spec.image_size).to(device)

    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    model.eval()
    return model


def cosine_loss(
    candidate: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> torch.Tensor:
    device = next(iter(candidate.values())).device
    dot = torch.zeros((), device=device)
    candidate_norm_sq = torch.zeros((), device=device)
    target_norm_sq = torch.zeros((), device=device)

    for name, current in candidate.items():
        reference = target[name].to(device=current.device, dtype=current.dtype)
        dot = dot + torch.sum(current * reference)
        candidate_norm_sq = candidate_norm_sq + torch.sum(current.square())
        target_norm_sq = target_norm_sq + torch.sum(reference.square())

    denominator = (
        torch.sqrt(candidate_norm_sq.clamp_min(1e-18))
        * torch.sqrt(target_norm_sq.clamp_min(1e-18))
    )
    return 1.0 - dot / denominator.clamp_min(1e-18)


def infer_idlg_label(
    decoded_observation: dict[str, torch.Tensor],
    num_classes: int,
) -> tuple[int, str]:
    bias_candidates: list[tuple[str, torch.Tensor]] = []
    weight_candidates: list[tuple[str, torch.Tensor]] = []

    for name, tensor in decoded_observation.items():
        value = tensor.detach().float().cpu()
        if value.ndim == 1 and value.numel() == num_classes and name.endswith("bias"):
            bias_candidates.append((name, value))
        if value.ndim >= 2 and value.shape[0] == num_classes and name.endswith("weight"):
            weight_candidates.append((name, value))

    if bias_candidates:
        name, gradient = sorted(bias_candidates, key=lambda x: x[0])[-1]
        return int(torch.argmin(gradient).item()), f"{name}:decoded_argmin_bias"

    if weight_candidates:
        name, gradient = sorted(weight_candidates, key=lambda x: x[0])[-1]
        scores = gradient.reshape(num_classes, -1).sum(dim=1)
        return int(torch.argmin(scores).item()), f"{name}:decoded_argmin_row_sum"

    raise RuntimeError("iDLG label inference failed on decoded Dense-RP observation.")


def run_attack(args) -> dict[str, object]:
    set_global_seed(args.seed, deterministic=True)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    spec = get_dataset_spec("cifar10")
    mean = spec.mean
    std = spec.std

    model = load_model(args.checkpoint, device)

    settings = DenseRPSettings(
        compression_ratio=args.compression_ratio,
        block_size=args.block_size,
        base_seed=args.projection_seed,
        round_index=args.round_index,
        dynamic=True,
        reconstruction_mode="projected",
        device=str(device),
        batch_blocks=args.projection_batch_blocks,
    )
    clear_dense_rp_cache()

    dataset = build_attack_dataset(
        dataset="cifar10",
        data_root=args.data_root,
        split="train",
        download=True,
    )
    image, true_label = dataset[args.sample_index]
    target_image = image.unsqueeze(0).to(device)
    target_label = torch.tensor([true_label], dtype=torch.long, device=device)

    raw_target = compute_named_gradients(
        model=model,
        image_01=target_image,
        label=target_label,
        parameter_scope="all",
        create_graph=False,
        mean=mean,
        std=std,
    )

    target_encoded = {
        name: encode_tensor(
            gradient.detach().cpu(),
            tensor_name=name,
            settings=settings,
        ).to(device=device, dtype=torch.float32)
        for name, gradient in raw_target.items()
    }

    decoded_target = None
    inferred_label = None
    label_source = "joint_optimization"

    if args.attack == "idlg":
        decoded_target = {
            name: decode_tensor(
                encoded=target_encoded[name].detach().cpu(),
                original_shape=gradient.shape,
                tensor_name=name,
                settings=settings,
            )
            for name, gradient in raw_target.items()
        }
        inferred_label, label_source = infer_idlg_label(decoded_target, 10)
        print(
            f"iDLG inferred label={inferred_label}, true={true_label}, "
            f"source={label_source}"
        )

    del raw_target
    if decoded_target is not None:
        del decoded_target

    attack_seed = args.seed + args.sample_index * 1009
    generator = torch.Generator(device=device)
    generator.manual_seed(attack_seed)

    initial = torch.rand((1, 3, 32, 32), generator=generator, device=device)
    initial = initial.clamp(1e-4, 1.0 - 1e-4)
    image_parameter = torch.logit(initial).detach().requires_grad_(True)

    label_logits = None
    parameter_groups = [{"params": [image_parameter], "lr": args.attack_lr}]
    if args.attack == "dlg":
        label_logits = torch.zeros(10, device=device, requires_grad=True)
        parameter_groups.append({"params": [label_logits], "lr": args.label_lr})

    optimizer = torch.optim.Adam(parameter_groups)

    hard_label_value = true_label if inferred_label is None else inferred_label
    hard_label = torch.tensor([hard_label_value], dtype=torch.long, device=device)

    progress_path = output_dir / f"progress_{args.attack}_dense_rp.csv"
    if progress_path.exists():
        progress_path.unlink()

    best_match = float("inf")
    best_iteration = 0
    best_label = hard_label_value
    best_image = torch.sigmoid(image_parameter.detach()).clone()
    final_match = float("nan")

    print("=" * 104)
    print(f"Dense-RP gradient inversion | attack={args.attack.upper()} | device={device}")
    print(f"checkpoint={args.checkpoint}")
    print(
        f"sample={args.sample_index} | ratio={args.compression_ratio} | "
        f"block={args.block_size} | projection_seed={args.projection_seed} | "
        f"round={args.round_index}"
    )
    print(
        f"iterations={args.iterations} | optimizer=Adam | loss=cosine | "
        f"attack_lr={args.attack_lr} | tv_weight={args.tv_weight}"
    )
    print("Attacker knows the Dense-RP mechanism/projection and directly matches encoded uploads.")
    print("=" * 104)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    for iteration in range(1, args.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        candidate_image = torch.sigmoid(image_parameter)

        raw_candidate = compute_named_gradients(
            model=model,
            image_01=candidate_image,
            label=hard_label if label_logits is None else None,
            parameter_scope="all",
            create_graph=True,
            mean=mean,
            std=std,
            label_logits=label_logits,
        )

        candidate_encoded = {
            name: differentiable_encode_tensor(
                gradient,
                tensor_name=name,
                settings=settings,
            )
            for name, gradient in raw_candidate.items()
        }

        match = cosine_loss(candidate_encoded, target_encoded)
        tv = total_variation(candidate_image)
        objective = match + args.tv_weight * tv

        if label_logits is not None and args.label_entropy_weight > 0:
            probabilities = torch.softmax(label_logits, dim=0)
            entropy = -(
                probabilities
                * torch.log(probabilities.clamp_min(1e-12))
            ).sum()
            objective = objective + args.label_entropy_weight * entropy

        attack_parameters = [image_parameter]
        if label_logits is not None:
            attack_parameters.append(label_logits)

        gradients = torch.autograd.grad(
            objective,
            attack_parameters,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )
        for parameter, gradient in zip(attack_parameters, gradients):
            parameter.grad = gradient

        optimizer.step()

        predicted_label = (
            hard_label_value
            if label_logits is None
            else int(torch.argmax(label_logits.detach()).item())
        )

        current_match = float(match.detach().item())
        if current_match < best_match:
            best_match = current_match
            best_iteration = iteration
            best_label = predicted_label
            best_image = candidate_image.detach().clone()

        final_match = current_match

        should_print = (
            iteration == 1
            or iteration % args.print_every == 0
            or iteration == args.iterations
        )
        if should_print:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            sec_per_iter = elapsed / iteration
            eta = sec_per_iter * max(args.iterations - iteration, 0)

            print(
                f"  attack={args.attack:4s} obs=encoded "
                f"iter={iteration:5d}/{args.iterations} | "
                f"match={current_match:.6e} | "
                f"label={predicted_label}/{true_label} | "
                f"{sec_per_iter:.3f}s/iter | ETA={eta / 60:.1f}m",
                flush=True,
            )
            append_csv(
                progress_path,
                {
                    "iteration": iteration,
                    "match_loss": current_match,
                    "predicted_label": predicted_label,
                    "best_match_loss": best_match,
                    "best_iteration": best_iteration,
                    "elapsed_seconds": elapsed,
                },
            )

        del raw_candidate, candidate_encoded, match, objective

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    attack_seconds = time.perf_counter() - start

    final_image = torch.sigmoid(image_parameter.detach())
    target_cpu = target_image.detach().cpu()
    best_cpu = best_image.detach().cpu()
    final_cpu = final_image.detach().cpu()

    save_image(target_cpu, output_dir / "target.png")
    save_image(best_cpu, output_dir / f"reconstruction_{args.attack}_dense_rp_best.png")
    save_image(final_cpu, output_dir / f"reconstruction_{args.attack}_dense_rp_final.png")

    optional_lpips = OptionalLPIPS(device, enabled=not args.no_lpips)
    lpips_value = optional_lpips(target_image, best_image.to(device))

    result = {
        "attack_name": args.attack,
        "observation_mode": "encoded",
        "sample_index": int(args.sample_index),
        "true_label": int(true_label),
        "predicted_label": int(best_label),
        "label_correct": int(best_label == true_label),
        "inferred_label": None if inferred_label is None else int(inferred_label),
        "label_source": label_source,
        "checkpoint": args.checkpoint,
        "compression_ratio": float(args.compression_ratio),
        "block_size": int(args.block_size),
        "projection_seed": int(args.projection_seed),
        "round_index": int(args.round_index),
        "iterations": int(args.iterations),
        "best_iteration": int(best_iteration),
        "best_match_loss": float(best_match),
        "final_match_loss": float(final_match),
        "mse": float(mse_01(target_cpu, best_cpu)),
        "psnr": float(psnr_01(target_cpu, best_cpu)),
        "ssim": float(ssim_01(target_cpu, best_cpu)),
        "lpips": None if lpips_value is None else float(lpips_value),
        "attack_seconds": float(attack_seconds),
    }

    (output_dir / "dense_rp_attack_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("-" * 104)
    print(
        f"sample={args.sample_index:05d} attack={args.attack.upper():4s} "
        f"obs=encoded | PSNR={result['psnr']:.2f} | "
        f"SSIM={result['ssim']:.4f} | label={best_label}/{true_label} | "
        f"best_iter={best_iteration} | {attack_seconds:.1f}s"
    )
    print(f"Artifacts saved to: {output_dir.resolve()}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="White-box DLG/iDLG attack against Dense-RP encoded uploads."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attack", choices=["dlg", "idlg"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--projection-seed", type=int, default=2026)
    parser.add_argument("--round-index", type=int, default=21)
    parser.add_argument("--sample-index", type=int, default=1)
    parser.add_argument("--compression-ratio", type=float, default=0.3)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--projection-batch-blocks", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--attack-lr", type=float, default=0.1)
    parser.add_argument("--label-lr", type=float, default=0.05)
    parser.add_argument("--tv-weight", type=float, default=1e-4)
    parser.add_argument("--label-entropy-weight", type=float, default=1e-3)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-lpips", action="store_true")
    return parser


if __name__ == "__main__":
    run_attack(build_parser().parse_args())
