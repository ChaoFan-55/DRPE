from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Mapping

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

from flocora_baseline.flocora import LoRAConv2d, inject_flocora


def append_csv(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_flocora_model(
    checkpoint_path: str,
    device: torch.device,
    rank: int,
    lora_alpha: float,
) -> torch.nn.Module:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    spec = get_dataset_spec("cifar10")
    model = build_model("resnet18", 10, "group", spec.image_size)
    inject_flocora(model, rank=rank, alpha=lora_alpha)
    model = model.to(device)

    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def factor_gradient_observation(
    gradients: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Direct gradient observation on FLoCoRA trainable parameters."""
    return dict(gradients)


def effective_update_observation(
    model: torch.nn.Module,
    gradients: Mapping[str, torch.Tensor],
    virtual_local_lr: float,
) -> dict[str, torch.Tensor]:
    """
    Decode FLoCoRA factor gradients into the exact dense effective convolution
    change induced by ONE plain-SGD virtual step:

        A1 = A - lr * g_A
        B1 = B - lr * g_B
        DeltaW_eff = (alpha/r) * (A1 B1 - A B)

    For trainable GroupNorm / final Linear parameters, the one-step observation
    is simply -lr * gradient.

    This is a gradient-level, representation-aware screening attack. It is NOT
    a replay of the historical 5-local-epoch client state upload.
    """
    if virtual_local_lr <= 0:
        raise ValueError("virtual_local_lr must be positive")

    observation: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()

    for module_name, module in model.named_modules():
        if not isinstance(module, LoRAConv2d):
            continue

        prefix = f"{module_name}." if module_name else ""
        a_name = prefix + "lora_up.weight"
        b_name = prefix + "lora_down.weight"
        if a_name not in gradients or b_name not in gradients:
            raise KeyError(
                f"Missing LoRA gradients for {module_name!r}: "
                f"expected {a_name!r} and {b_name!r}."
            )

        # Treat the known checkpoint factors as constants for the outer attack.
        a = module.lora_up.weight.detach()[:, :, 0, 0]  # [O, r]
        b = module.lora_down.weight.detach()            # [r, I, K, K]
        g_a = gradients[a_name][:, :, 0, 0]
        g_b = gradients[b_name]

        b_flat = b.reshape(b.shape[0], -1)
        g_b_flat = g_b.reshape(g_b.shape[0], -1)

        a1 = a - virtual_local_lr * g_a
        b1_flat = b_flat - virtual_local_lr * g_b_flat

        before = a @ b_flat
        after = a1 @ b1_flat
        delta = (after - before) * module.scaling
        delta = delta.reshape_as(module.base.weight)

        observation[prefix + "effective_delta_weight"] = delta
        consumed.add(a_name)
        consumed.add(b_name)

    # GroupNorm and final FC are trained/transmitted directly in this baseline.
    # Their data-dependent one-step update is -lr*g. (Known deterministic
    # optimizer regularization is intentionally not used as privacy signal.)
    for name, gradient in gradients.items():
        if name in consumed:
            continue
        observation[name] = -virtual_local_lr * gradient

    if not observation:
        raise ValueError("Empty FLoCoRA effective-update observation.")
    return observation


def cosine_matching_loss(
    candidate: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    if candidate.keys() != target.keys():
        missing = sorted(set(target) - set(candidate))
        extra = sorted(set(candidate) - set(target))
        raise ValueError(f"Observation key mismatch; missing={missing}, extra={extra}")

    first = next(iter(candidate.values()), None)
    if first is None:
        raise ValueError("Empty FLoCoRA observation.")

    device = first.device
    dot = torch.zeros((), device=device)
    cand_norm_sq = torch.zeros((), device=device)
    targ_norm_sq = torch.zeros((), device=device)

    for name, current in candidate.items():
        reference = target[name].to(device=current.device, dtype=current.dtype)
        current_flat = current.reshape(-1)
        reference_flat = reference.reshape(-1)
        if current_flat.numel() != reference_flat.numel():
            raise ValueError(
                f"Observation length mismatch for {name}: "
                f"candidate={current_flat.numel()}, target={reference_flat.numel()}"
            )
        dot = dot + torch.sum(current_flat * reference_flat)
        cand_norm_sq = cand_norm_sq + torch.sum(current_flat.square())
        targ_norm_sq = targ_norm_sq + torch.sum(reference_flat.square())

    denominator = (
        torch.sqrt(cand_norm_sq.clamp_min(1e-18))
        * torch.sqrt(targ_norm_sq.clamp_min(1e-18))
    )
    return 1.0 - dot / denominator.clamp_min(1e-18)


def infer_idlg_label(
    target_observation: Mapping[str, torch.Tensor],
    observation_mode: str,
    virtual_local_lr: float,
    num_classes: int = 10,
) -> tuple[int, str]:
    """Infer iDLG label from the directly trained final classifier."""
    bias_candidates: list[tuple[str, torch.Tensor]] = []
    weight_candidates: list[tuple[str, torch.Tensor]] = []

    for name, value in target_observation.items():
        tensor = value.detach().float()
        # effective mode stores one-step delta=-lr*g for direct parameters;
        # convert it back to the gradient before standard iDLG inference.
        if observation_mode == "effective" and not name.endswith("effective_delta_weight"):
            tensor = -tensor / virtual_local_lr

        if tensor.ndim == 1 and tensor.numel() == num_classes and name.endswith("bias"):
            bias_candidates.append((name, tensor))
        if tensor.ndim >= 2 and tensor.shape[0] == num_classes and name.endswith("weight"):
            weight_candidates.append((name, tensor))

    if bias_candidates:
        name, gradient = sorted(bias_candidates, key=lambda item: item[0])[-1]
        return int(torch.argmin(gradient).item()), f"{name}:argmin_bias"

    if weight_candidates:
        name, gradient = sorted(weight_candidates, key=lambda item: item[0])[-1]
        scores = gradient.reshape(num_classes, -1).sum(dim=1)
        return int(torch.argmin(scores).item()), f"{name}:argmin_row_sum"

    raise RuntimeError("iDLG label inference failed on FLoCoRA observation.")


def make_observation(
    model: torch.nn.Module,
    gradients: Mapping[str, torch.Tensor],
    observation_mode: str,
    virtual_local_lr: float,
) -> dict[str, torch.Tensor]:
    if observation_mode == "factor":
        return factor_gradient_observation(gradients)
    if observation_mode == "effective":
        return effective_update_observation(
            model=model,
            gradients=gradients,
            virtual_local_lr=virtual_local_lr,
        )
    raise ValueError(f"Unsupported observation mode: {observation_mode}")


def run_attack(args) -> dict[str, object]:
    set_global_seed(args.seed, deterministic=True)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    spec = get_dataset_spec("cifar10")
    mean = spec.mean
    std = spec.std

    model = load_flocora_model(
        checkpoint_path=args.checkpoint,
        device=device,
        rank=args.rank,
        lora_alpha=args.lora_alpha,
    )

    dataset = build_attack_dataset(
        dataset="cifar10",
        data_root=args.data_root,
        split="train",
        download=True,
    )
    image, true_label = dataset[args.sample_index]
    target_image = image.unsqueeze(0).to(device)
    target_label = torch.tensor([true_label], dtype=torch.long, device=device)

    # Gradient-level simulated upload at the chosen checkpoint/sample.
    raw_target = compute_named_gradients(
        model=model,
        image_01=target_image,
        label=target_label,
        parameter_scope="all",
        create_graph=False,
        mean=mean,
        std=std,
    )
    target_observation = {
        name: value.detach()
        for name, value in make_observation(
            model=model,
            gradients=raw_target,
            observation_mode=args.observation,
            virtual_local_lr=args.virtual_local_lr,
        ).items()
    }

    inferred_label = None
    label_source = "joint_optimization"
    if args.attack == "idlg":
        inferred_label, label_source = infer_idlg_label(
            target_observation=target_observation,
            observation_mode=args.observation,
            virtual_local_lr=args.virtual_local_lr,
            num_classes=10,
        )
        print(
            f"iDLG inferred label={inferred_label}, true={true_label}, "
            f"source={label_source}"
        )

    # Privacy hygiene: the optimizer below only sees target_observation.
    del raw_target

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
    scheduler = None
    if args.cosine_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.iterations,
            eta_min=args.attack_lr * 0.01,
        )

    hard_label_value = true_label if inferred_label is None else inferred_label
    hard_label = torch.tensor([hard_label_value], dtype=torch.long, device=device)

    obs_tag = "effective_update" if args.observation == "effective" else "factor_grad"
    progress_path = output_dir / f"progress_{args.attack}_flocora_{obs_tag}.csv"
    if progress_path.exists():
        progress_path.unlink()

    best_match = float("inf")
    best_iteration = 0
    best_label = hard_label_value
    best_image = torch.sigmoid(image_parameter.detach()).clone()
    final_match = float("nan")

    print("=" * 118)
    print(
        f"FLoCoRA gradient inversion | attack={args.attack.upper()} | "
        f"obs={obs_tag} | device={device}"
    )
    print(f"checkpoint={args.checkpoint}")
    print(
        f"sample={args.sample_index} | rank={args.rank} | "
        f"lora_alpha={args.lora_alpha:g} | scaling={args.lora_alpha/args.rank:.3f} | "
        f"round={args.round_index}"
    )
    print(
        f"iterations={args.iterations} | optimizer=Adam | cosine_schedule={args.cosine_schedule} | "
        f"attack_lr={args.attack_lr} | tv_weight={args.tv_weight}"
    )
    if args.observation == "effective":
        print(
            "Target is a representation-aware one-step effective FLoCoRA update: "
            "LoRA factor gradients are decoded through the known checkpoint A/B factors; "
            "GroupNorm/head updates are matched directly. Raw target gradients are discarded "
            "before inversion optimization."
        )
    else:
        print(
            "Target directly matches gradients of the trainable FLoCoRA factors, "
            "GroupNorm, and final FC. Raw target image/data are not used by the optimizer."
        )
    print(
        "Protocol note: this is a single-example gradient-level simulated upload, not a replay "
        "of the historical 5-local-epoch client parameter state."
    )
    print("=" * 118)

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
        candidate_observation = make_observation(
            model=model,
            gradients=raw_candidate,
            observation_mode=args.observation,
            virtual_local_lr=args.virtual_local_lr,
        )

        match = cosine_matching_loss(candidate_observation, target_observation)
        tv = total_variation(candidate_image)
        objective = match + args.tv_weight * tv

        if label_logits is not None and args.label_entropy_weight > 0:
            probabilities = torch.softmax(label_logits, dim=0)
            entropy = -(
                probabilities * torch.log(probabilities.clamp_min(1e-12))
            ).sum()
            objective = objective + args.label_entropy_weight * entropy

        attack_parameters = [image_parameter]
        if label_logits is not None:
            attack_parameters.append(label_logits)

        outer_gradients = torch.autograd.grad(
            objective,
            attack_parameters,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )
        for parameter, gradient in zip(attack_parameters, outer_gradients):
            parameter.grad = gradient

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

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
                f"  attack={args.attack:4s} obs={obs_tag:16s} "
                f"iter={iteration:5d}/{args.iterations} | "
                f"match={current_match:.6e} | "
                f"label={predicted_label}/{true_label} | "
                f"{sec_per_iter:.3f}s/iter | ETA={eta/60:.1f}m",
                flush=True,
            )
            append_csv(
                progress_path,
                {
                    "iteration": iteration,
                    "observation": obs_tag,
                    "match_loss": current_match,
                    "predicted_label": predicted_label,
                    "best_match_loss": best_match,
                    "best_iteration": best_iteration,
                    "elapsed_seconds": elapsed,
                },
            )

        del raw_candidate, candidate_observation, match, objective

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    attack_seconds = time.perf_counter() - start

    final_image = torch.sigmoid(image_parameter.detach())
    target_cpu = target_image.detach().cpu()
    best_cpu = best_image.detach().cpu()
    final_cpu = final_image.detach().cpu()

    save_image(target_cpu, output_dir / "target.png")
    save_image(best_cpu, output_dir / f"reconstruction_{args.attack}_flocora_{obs_tag}_best.png")
    save_image(final_cpu, output_dir / f"reconstruction_{args.attack}_flocora_{obs_tag}_final.png")

    optional_lpips = OptionalLPIPS(device, enabled=not args.no_lpips)
    lpips_value = optional_lpips(target_image, best_image.to(device))

    result = {
        "attack_name": args.attack,
        "observation_mode": f"flocora_{obs_tag}",
        "protocol": "single_example_gradient_level_simulated_upload",
        "sample_index": int(args.sample_index),
        "true_label": int(true_label),
        "predicted_label": int(best_label),
        "label_correct": int(best_label == true_label),
        "inferred_label": None if inferred_label is None else int(inferred_label),
        "label_source": label_source,
        "checkpoint": args.checkpoint,
        "rank": int(args.rank),
        "lora_alpha": float(args.lora_alpha),
        "lora_scaling": float(args.lora_alpha / args.rank),
        "virtual_local_lr": float(args.virtual_local_lr),
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

    summary_path = output_dir / f"flocora_attack_{obs_tag}_summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("-" * 118)
    print(
        f"sample={args.sample_index:05d} attack={args.attack.upper():4s} "
        f"obs={obs_tag} | PSNR={result['psnr']:.2f} | "
        f"SSIM={result['ssim']:.4f} | label={best_label}/{true_label} | "
        f"best_match={best_match:.6e} | best_iter={best_iteration} | "
        f"{attack_seconds:.1f}s"
    )
    print(f"Artifacts saved to: {output_dir.resolve()}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "White-box DLG/iDLG attack against the trainable FLoCoRA parameter support. "
            "The main protocol directly matches LoRA-factor/GroupNorm/FC gradients; "
            "an effective-update view is available only as supplementary analysis."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attack", choices=["dlg", "idlg"], required=True)
    parser.add_argument(
        "--observation",
        choices=["factor", "effective"],
        default="factor",
        help=(
            "factor=direct matching on the actually trainable/uploaded FLoCoRA parameter support "
            "(main protocol); effective=attacker-derived one-step dense effective convolution update "
            "(supplementary analysis)"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--round-index", type=int, default=21)
    parser.add_argument("--sample-index", type=int, default=1)
    parser.add_argument("--rank", type=int, default=93)
    parser.add_argument("--lora-alpha", type=float, default=1488.0)
    parser.add_argument("--virtual-local-lr", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--attack-lr", type=float, default=0.1)
    parser.add_argument("--label-lr", type=float, default=0.05)
    parser.add_argument("--tv-weight", type=float, default=1e-4)
    parser.add_argument("--label-entropy-weight", type=float, default=1e-3)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument(
        "--no-cosine-schedule",
        dest="cosine_schedule",
        action="store_false",
        help="Disable cosine decay of the attack optimizer learning rate.",
    )
    parser.set_defaults(cosine_schedule=True)
    return parser


if __name__ == "__main__":
    run_attack(build_parser().parse_args())
