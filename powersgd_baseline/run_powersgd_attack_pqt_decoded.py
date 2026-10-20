from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

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

from powersgd_baseline.powersgd import (
    PowerSGDSettings,
    clear_powersgd_cache,
    decoded_payload_on_device,
    differentiable_encode_tensor,
    encode_tensor,
    target_payload_to_device,
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


def decode_target_payloads(
    target_payload: dict[str, dict[str, Any]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Reconstruct exactly what the server can reconstruct from intercepted
    PowerSGD payloads:
        dense tensor                  -> g
        low-rank transmitted (P, Q)   -> P Q^T
    No raw target gradient is retained by the attack after this step.
    """
    return {
        name: decoded_payload_on_device(payload, device).detach()
        for name, payload in target_payload.items()
    }


def infer_idlg_label_from_decoded(
    target_decoded: dict[str, torch.Tensor],
    num_classes: int,
) -> tuple[int, str]:
    bias_candidates: list[tuple[str, torch.Tensor]] = []
    weight_candidates: list[tuple[str, torch.Tensor]] = []

    for name, gradient in target_decoded.items():
        value = gradient.float()
        if value.ndim == 1 and value.numel() == num_classes and name.endswith("bias"):
            bias_candidates.append((name, value))
        if value.ndim >= 2 and value.shape[0] == num_classes and name.endswith("weight"):
            weight_candidates.append((name, value))

    if bias_candidates:
        name, gradient = sorted(bias_candidates, key=lambda item: item[0])[-1]
        return int(torch.argmin(gradient).item()), f"{name}:decoded_argmin_bias"

    if weight_candidates:
        name, gradient = sorted(weight_candidates, key=lambda item: item[0])[-1]
        scores = gradient.reshape(num_classes, -1).sum(dim=1)
        return int(torch.argmin(scores).item()), f"{name}:decoded_argmin_row_sum"

    raise RuntimeError("iDLG label inference failed on decoded PowerSGD observation.")


def _accumulate_cosine_terms(
    current: torch.Tensor,
    reference: torch.Tensor,
    dot: torch.Tensor,
    current_norm_sq: torch.Tensor,
    reference_norm_sq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    current_flat = current.reshape(-1)
    reference_flat = reference.to(
        device=current_flat.device,
        dtype=current_flat.dtype,
    ).reshape(-1)

    if current_flat.numel() != reference_flat.numel():
        raise ValueError(
            "Decoded PowerSGD tensor length mismatch: "
            f"candidate={current_flat.numel()}, target={reference_flat.numel()}."
        )

    dot = dot + torch.sum(current_flat * reference_flat)
    current_norm_sq = current_norm_sq + torch.sum(current_flat.square())
    reference_norm_sq = reference_norm_sq + torch.sum(reference_flat.square())
    return dot, current_norm_sq, reference_norm_sq


def decoded_product_cosine_loss(
    raw_candidate: dict[str, torch.Tensor],
    target_decoded: dict[str, torch.Tensor],
    settings: PowerSGDSettings,
) -> torch.Tensor:
    """
    Compressor-aware PowerSGD inversion loss.

    Target for every low-rank tensor:
        G_hat_t = P_t Q_t^T

    Candidate for every low-rank tensor:
        G_hat_c = P_c Q_c^T

    where (P_c, Q_c) are produced by the same differentiable PowerSGD
    compressor from the candidate gradient. Dense-fallback tensors are
    compared directly because their transmitted value is already dense.

    This deliberately avoids matching P and Q separately, because factor
    pairs are not a unique representation of the same decoded product.
    """

    first_gradient = next(iter(raw_candidate.values()), None)
    if first_gradient is None:
        raise ValueError("Empty candidate gradient set.")

    device = first_gradient.device
    dot = torch.zeros((), device=device)
    current_norm_sq = torch.zeros((), device=device)
    reference_norm_sq = torch.zeros((), device=device)

    for name, gradient in raw_candidate.items():
        reference = target_decoded[name]

        payload = differentiable_encode_tensor(
            gradient,
            tensor_name=name,
            settings=settings,
        )
        if payload["kind"] == "dense":
            current = payload["value"]
        else:
            current = (
                payload["p"] @ payload["q"].transpose(0, 1)
            ).reshape(payload["shape"])

        dot, current_norm_sq, reference_norm_sq = _accumulate_cosine_terms(
            current=current,
            reference=reference,
            dot=dot,
            current_norm_sq=current_norm_sq,
            reference_norm_sq=reference_norm_sq,
        )

    denominator = (
        torch.sqrt(current_norm_sq.clamp_min(1e-18))
        * torch.sqrt(reference_norm_sq.clamp_min(1e-18))
    )
    return 1.0 - dot / denominator.clamp_min(1e-18)


def run_attack(args) -> dict[str, object]:
    set_global_seed(args.seed, deterministic=True)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    spec = get_dataset_spec("cifar10")
    mean = spec.mean
    std = spec.std

    model = load_model(args.checkpoint, device)

    settings = PowerSGDSettings(
        rank=args.rank,
        base_seed=args.powersgd_seed,
        round_index=args.round_index,
        client_id=args.client_id,
        power_iterations=args.power_iterations,
        device=str(device),
        fallback_dense_if_not_smaller=True,
    )
    clear_powersgd_cache()

    dataset = build_attack_dataset(
        dataset="cifar10",
        data_root=args.data_root,
        split="train",
        download=True,
    )
    image, true_label = dataset[args.sample_index]
    target_image = image.unsqueeze(0).to(device)
    target_label = torch.tensor([true_label], dtype=torch.long, device=device)

    # Build the target transmission from the true single-example gradient,
    # then immediately discard the raw target after deriving the transmitted
    # payload and the server-equivalent decoded observation.
    raw_target = compute_named_gradients(
        model=model,
        image_01=target_image,
        label=target_label,
        parameter_scope="all",
        create_graph=False,
        mean=mean,
        std=std,
    )

    target_payload_cpu = {
        name: encode_tensor(
            gradient.detach().cpu(),
            tensor_name=name,
            settings=settings,
        )
        for name, gradient in raw_target.items()
    }
    target_payload = {
        name: target_payload_to_device(payload, device)
        for name, payload in target_payload_cpu.items()
    }
    target_decoded = decode_target_payloads(target_payload, device)

    inferred_label = None
    label_source = "joint_optimization"

    if args.attack == "idlg":
        inferred_label, label_source = infer_idlg_label_from_decoded(
            target_decoded=target_decoded,
            num_classes=10,
        )
        print(
            f"iDLG inferred label={inferred_label}, true={true_label}, "
            f"source={label_source}"
        )

    # Critical privacy hygiene for the attack implementation:
    # after this point, the optimization has no access to raw_target.
    del raw_target, target_payload_cpu

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

    progress_path = output_dir / (
        f"progress_{args.attack}_powersgd_pqt_decoded.csv"
    )
    if progress_path.exists():
        progress_path.unlink()

    best_match = float("inf")
    best_iteration = 0
    best_label = hard_label_value
    best_image = torch.sigmoid(image_parameter.detach()).clone()
    final_match = float("nan")

    print("=" * 112)
    print(
        f"PowerSGD PQt decoded-product inversion | attack={args.attack.upper()} | "
        f"device={device}"
    )
    print(f"checkpoint={args.checkpoint}")
    print(
        f"sample={args.sample_index} | rank={args.rank} | "
        f"power_iterations={args.power_iterations} | "
        f"powersgd_seed={args.powersgd_seed} | round={args.round_index} | "
        f"client={args.client_id}"
    )
    print(
        f"iterations={args.iterations} | optimizer=Adam | loss=cosine | "
        f"attack_lr={args.attack_lr} | tv_weight={args.tv_weight}"
    )

    print(
        "Target=P_t Q_t^T from intercepted payload; candidate gradient is passed "
        "through the same differentiable PowerSGD compressor and decoded as "
        "P_c Q_c^T before cosine matching. Raw target gradients are not used "
        "during optimization."
    )
    print("=" * 112)

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

        match = decoded_product_cosine_loss(
            raw_candidate=raw_candidate,
            target_decoded=target_decoded,
            settings=settings,
        )
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
                f"  attack={args.attack:4s} obs=pqt_decoded    "
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
                    "observation": "pqt_decoded_product",
                    "match_loss": current_match,
                    "predicted_label": predicted_label,
                    "best_match_loss": best_match,
                    "best_iteration": best_iteration,
                    "elapsed_seconds": elapsed,
                },
            )

        del raw_candidate, match, objective

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    attack_seconds = time.perf_counter() - start

    final_image = torch.sigmoid(image_parameter.detach())
    target_cpu = target_image.detach().cpu()
    best_cpu = best_image.detach().cpu()
    final_cpu = final_image.detach().cpu()

    save_image(target_cpu, output_dir / "target.png")
    save_image(
        best_cpu,
        output_dir
        / f"reconstruction_{args.attack}_powersgd_pqt_decoded_best.png",
    )
    save_image(
        final_cpu,
        output_dir
        / f"reconstruction_{args.attack}_powersgd_pqt_decoded_final.png",
    )

    optional_lpips = OptionalLPIPS(device, enabled=not args.no_lpips)
    lpips_value = optional_lpips(target_image, best_image.to(device))

    result = {
        "attack_name": args.attack,
        "observation_mode": "pqt_decoded_product",
        "target_source": "intercepted_powersgd_payload_decoded_as_PQt",
        "raw_target_used_during_optimization": False,
        "sample_index": int(args.sample_index),
        "true_label": int(true_label),
        "predicted_label": int(best_label),
        "label_correct": int(best_label == true_label),
        "inferred_label": None if inferred_label is None else int(inferred_label),
        "label_source": label_source,
        "checkpoint": args.checkpoint,
        "rank": int(args.rank),
        "power_iterations": int(args.power_iterations),
        "powersgd_seed": int(args.powersgd_seed),
        "round_index": int(args.round_index),
        "client_id": int(args.client_id),
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

    summary_path = (
        output_dir
        / "powersgd_pqt_decoded_attack_summary.json"
    )
    summary_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("-" * 112)
    print(
        f"sample={args.sample_index:05d} attack={args.attack.upper():4s} "
        f"obs=pqt_decoded | PSNR={result['psnr']:.2f} | "
        f"SSIM={result['ssim']:.4f} | label={best_label}/{true_label} | "
        f"best_match={best_match:.6e} | best_iter={best_iteration} | "
        f"{attack_seconds:.1f}s"
    )
    print(f"Artifacts saved to: {output_dir.resolve()}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "White-box DLG/iDLG attack using PowerSGD P Q^T decoded-product matching."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attack", choices=["dlg", "idlg"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--powersgd-seed", type=int, default=2026)
    parser.add_argument("--round-index", type=int, default=21)
    parser.add_argument("--client-id", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=1)
    parser.add_argument("--rank", type=int, default=97)
    parser.add_argument("--power-iterations", type=int, default=1)
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
