from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from ..compression import decode_update_dict, encode_update_dict
from ..config import model_encoded_numel
from ..data import build_attack_dataset, get_dataset_spec
from ..models import build_model
from ..projection import ProjectionSettings
from ..utils import ensure_dir, resolve_device, set_global_seed
from .classic_config import ClassicAttackConfig
from .inversion import compute_named_gradients, total_variation
from .metrics import OptionalLPIPS, mse_01, psnr_01, ssim_01
from .operators import differentiable_decode_dict, differentiable_encode_dict


@dataclass
class ClassicAttackResult:
    best_image: torch.Tensor
    final_image: torch.Tensor
    predicted_label: int
    inferred_label: int | None
    label_source: str
    best_iteration: int
    best_objective: float
    best_match_loss: float
    final_objective: float
    final_match_loss: float


def _make_projection_settings(cfg: ClassicAttackConfig) -> ProjectionSettings:
    return ProjectionSettings(
        compression_ratio=cfg.compression_ratio,
        block_size=cfg.block_size,
        base_seed=cfg.projection_seed,
        round_index=int(cfg.projection_round or 0),
        dynamic=cfg.dynamic_projection,
        reconstruction_mode=cfg.reconstruction_mode,
        device=cfg.projection_device,
    )


def _load_model(cfg: ClassicAttackConfig, device: torch.device) -> nn.Module:
    spec = get_dataset_spec(cfg.dataset)
    model = build_model(cfg.model_name, cfg.num_classes, cfg.norm, spec.image_size).to(device)
    if cfg.checkpoint_path is not None:
        checkpoint = Path(cfg.checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Attack checkpoint does not exist: {checkpoint}. "
                "Use an existing checkpoint_round_XXXX.pt/best_model.pt path."
            )
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)
    model.eval()
    return model


def _build_target_dataset(cfg: ClassicAttackConfig):
    return build_attack_dataset(
        dataset=cfg.dataset,
        data_root=cfg.data_root,
        split=cfg.target_split,
        download=cfg.dataset_download,
    )


def _resolve_attack_parameters(cfg: ClassicAttackConfig, model: nn.Module) -> None:
    if cfg.observation_mode == "raw":
        cfg.observation_clip_norm = None
        cfg.noise_std = 0.0
        return
    if cfg.projection_round is None:
        if not cfg.training_summary_path:
            raise ValueError("DRPE attack requires projection_round or training_summary_path.")
        summary_path = Path(cfg.training_summary_path)
        if not summary_path.exists():
            raise FileNotFoundError(f"Training summary not found: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cfg.projection_round = int(float(summary["best_round"])) + 1
    if cfg.calibration_path:
        calibration_path = Path(cfg.calibration_path)
        if not calibration_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if cfg.observation_clip_norm is None:
            cfg.observation_clip_norm = float(calibration.get("selected_clip_norm", calibration["p75"]))
    if cfg.observation_clip_norm is None:
        raise ValueError("DRPE attack requires observation_clip_norm or calibration_path.")
    if cfg.noise_std is None:
        count = model_encoded_numel(model, cfg.block_size, cfg.compression_ratio)
        cfg.noise_std = float(cfg.noise_eta * cfg.observation_clip_norm / np.sqrt(count))


def _matching_loss(
    candidate: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    kind: str,
    relative_l2_floor: float = 1e-8,
) -> torch.Tensor:
    if candidate.keys() != target.keys():
        raise ValueError("Candidate and target observations must have identical keys.")

    device = next(iter(candidate.values())).device
    if kind in {"l2_sum", "l2_mean"}:
        total = torch.zeros((), device=device)
        count = 0
        for name, current in candidate.items():
            reference = target[name].to(device=current.device, dtype=current.dtype)
            difference = current - reference
            total = total + torch.sum(difference.square())
            count += difference.numel()
        if kind == "l2_mean":
            total = total / max(count, 1)
        return total

    if kind == "relative_l2":
        # Average a scale-normalized loss over parameter tensors. This avoids
        # the largest convolution layers dominating the full-model objective.
        layer_losses: list[torch.Tensor] = []
        for name, current in candidate.items():
            reference = target[name].to(device=current.device, dtype=current.dtype)
            difference_energy = torch.mean((current - reference).square())
            reference_energy = torch.mean(reference.square())
            denominator = reference_energy + current.new_tensor(relative_l2_floor)
            layer_losses.append(difference_energy / denominator)
        return torch.stack(layer_losses).mean()

    if kind == "cosine":
        dot = torch.zeros((), device=device)
        candidate_norm_sq = torch.zeros_like(dot)
        target_norm_sq = torch.zeros_like(dot)
        for name, current in candidate.items():
            reference = target[name].to(device=current.device, dtype=current.dtype)
            dot = dot + torch.sum(current * reference)
            candidate_norm_sq = candidate_norm_sq + torch.sum(current.square())
            target_norm_sq = target_norm_sq + torch.sum(reference.square())
        denominator = torch.sqrt(candidate_norm_sq.clamp_min(1e-18)) * torch.sqrt(
            target_norm_sq.clamp_min(1e-18)
        )
        return 1.0 - dot / denominator.clamp_min(1e-18)

    raise ValueError(f"Unsupported matching loss: {kind}")


def infer_idlg_label(
    observation: Mapping[str, torch.Tensor],
    num_classes: int,
) -> tuple[int, str]:
    """Infer a batch-size-one label from the final linear-layer gradient.

    Prefer the classification bias gradient because its true-class entry is
    negative for cross-entropy with a single sample. Fall back to row sums of
    the final weight gradient when the bias is absent.
    """
    bias_candidates: list[tuple[str, torch.Tensor]] = []
    weight_candidates: list[tuple[str, torch.Tensor]] = []

    for name, tensor in observation.items():
        value = tensor.detach().float().cpu()
        if value.ndim == 1 and value.numel() == num_classes and name.endswith("bias"):
            bias_candidates.append((name, value))
        if value.ndim >= 2 and value.shape[0] == num_classes and name.endswith("weight"):
            weight_candidates.append((name, value))

    def _rank(name: str) -> tuple[int, int]:
        preferred = int(not (name.endswith("head.bias") or name.endswith("head.weight")))
        return preferred, len(name)

    if bias_candidates:
        name, gradient = sorted(bias_candidates, key=lambda item: _rank(item[0]))[0]
        return int(torch.argmin(gradient).item()), f"{name}:argmin_bias"

    if weight_candidates:
        name, gradient = sorted(weight_candidates, key=lambda item: _rank(item[0]))[0]
        row_scores = gradient.reshape(num_classes, -1).sum(dim=1)
        return int(torch.argmin(row_scores).item()), f"{name}:argmin_row_sum"

    raise ValueError(
        "iDLG label inference could not find the final classifier bias/weight "
        "gradient in the selected parameter scope. Use parameter_scope='all' "
        "or 'last_layer'."
    )



def _differentiable_global_l2_clip(
    tensors: Mapping[str, torch.Tensor],
    clip_norm: float | None,
) -> Dict[str, torch.Tensor]:
    """Apply one global L2 clipping factor while preserving autograd."""
    if clip_norm is None:
        return dict(tensors)
    if not tensors:
        return {}
    device = next(iter(tensors.values())).device
    norm_sq = torch.zeros((), device=device)
    for tensor in tensors.values():
        norm_sq = norm_sq + torch.sum(tensor.square())
    norm = torch.sqrt(norm_sq.clamp_min(1e-24))
    scale = torch.clamp(norm.new_tensor(float(clip_norm)) / norm, max=1.0)
    return {name: tensor * scale for name, tensor in tensors.items()}

def _candidate_observation(
    raw_gradients: Mapping[str, torch.Tensor],
    cfg: ClassicAttackConfig,
    settings: ProjectionSettings | None,
    original_shapes: Mapping[str, torch.Size],
) -> Dict[str, torch.Tensor]:
    clipped = _differentiable_global_l2_clip(
        raw_gradients, cfg.observation_clip_norm
    )
    if cfg.observation_mode == "raw":
        return clipped
    if settings is None:
        raise ValueError(
            "Projection settings are required for decoded/encoded observations."
        )
    encoded = differentiable_encode_dict(
        clipped,
        settings,
        batch_blocks=cfg.projection_batch_blocks,
    )
    if cfg.observation_mode == "encoded":
        return encoded
    return differentiable_decode_dict(
        encoded,
        original_shapes,
        settings,
        batch_blocks=cfg.projection_batch_blocks,
    )


def _target_observation(
    raw_gradients: Mapping[str, torch.Tensor],
    cfg: ClassicAttackConfig,
    settings: ProjectionSettings | None,
) -> Dict[str, torch.Tensor]:
    clipped = _differentiable_global_l2_clip(
        raw_gradients, cfg.observation_clip_norm
    )
    if cfg.observation_mode == "raw":
        return {name: tensor.detach().float().cpu() for name, tensor in clipped.items()}
    if settings is None:
        raise ValueError(
            "Projection settings are required for decoded/encoded observations."
        )

    if cfg.noise_std == 0.0:
        original_shapes = {name: tensor.shape for name, tensor in clipped.items()}
        observation = _candidate_observation(clipped, cfg, settings, original_shapes)
        return {
            name: tensor.detach().float().cpu()
            for name, tensor in observation.items()
        }

    clipped_cpu = {
        name: tensor.detach().float().cpu() for name, tensor in clipped.items()
    }
    encoded = encode_update_dict(
        updates=clipped_cpu,
        settings=settings,
        noise_std=cfg.noise_std,
        client_id=cfg.observed_client_id,
        noise_seed=cfg.noise_seed,
    )
    if cfg.observation_mode == "encoded":
        return {
            name: tensor.detach().float().cpu()
            for name, tensor in encoded.tensors.items()
        }
    decoded = decode_update_dict(
        encoded=encoded.tensors,
        original_shapes=encoded.original_shapes,
        settings=settings,
    )
    return {
        name: tensor.detach().float().cpu() for name, tensor in decoded.items()
    }


def _append_csv(path: Path, row: Mapping[str, object]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _initialize_image_parameter(
    cfg: ClassicAttackConfig,
    device: torch.device,
    image_size: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)
    if cfg.image_parameterization == "sigmoid":
        if cfg.init == "normal":
            parameter = torch.randn((1, 3, image_size, image_size), generator=generator, device=device)
        else:
            initial = torch.rand((1, 3, image_size, image_size), generator=generator, device=device)
            parameter = torch.logit(initial.clamp(1e-4, 1 - 1e-4))
    else:
        if cfg.init == "normal":
            parameter = 0.5 + 0.25 * torch.randn(
                (1, 3, image_size, image_size), generator=generator, device=device
            )
        else:
            parameter = torch.rand((1, 3, image_size, image_size), generator=generator, device=device)
        parameter = parameter.clamp(0, 1)
    return parameter.requires_grad_(True)


def _image_from_parameter(
    image_parameter: torch.Tensor,
    parameterization: str,
) -> torch.Tensor:
    if parameterization == "sigmoid":
        return torch.sigmoid(image_parameter)
    return image_parameter.clamp(0, 1)


def _run_single_attack(
    model: nn.Module,
    target_observation: Mapping[str, torch.Tensor],
    true_label: int,
    cfg: ClassicAttackConfig,
    settings: ProjectionSettings | None,
    original_shapes: Mapping[str, torch.Size],
    device: torch.device,
    progress_path: Path,
) -> ClassicAttackResult:
    target_on_device = {
        name: tensor.detach().to(device=device, dtype=torch.float32)
        for name, tensor in target_observation.items()
    }

    inferred_label: int | None = None
    label_source = (
        "oracle_known" if cfg.label_mode == "known" else "joint_optimization"
    )
    if cfg.attack_name == "idlg" and cfg.label_mode != "known":
        inferred_label, label_source = infer_idlg_label(
            target_observation, cfg.num_classes
        )

    spec = get_dataset_spec(cfg.dataset)
    image_parameter = _initialize_image_parameter(cfg, device, spec.image_size)
    label_logits: torch.Tensor | None = None
    attack_parameters: list[torch.Tensor] = [image_parameter]
    if cfg.attack_name == "dlg" and cfg.label_mode == "optimize":
        if cfg.label_init == "zeros":
            label_logits = torch.zeros(cfg.num_classes, device=device, requires_grad=True)
        else:
            label_generator = torch.Generator(device=device)
            label_generator.manual_seed(cfg.seed + 7919)
            label_logits = torch.randn(
                cfg.num_classes,
                generator=label_generator,
                device=device,
                requires_grad=True,
            )
        attack_parameters.append(label_logits)

    if cfg.optimizer == "adam":
        parameter_groups: list[dict[str, object]] = [
            {"params": [image_parameter], "lr": cfg.attack_lr}
        ]
        if label_logits is not None:
            parameter_groups.append(
                {"params": [label_logits], "lr": cfg.label_lr or cfg.attack_lr}
            )
        optimizer: torch.optim.Optimizer = torch.optim.Adam(parameter_groups)
    else:
        optimizer = torch.optim.LBFGS(
            attack_parameters,
            lr=cfg.attack_lr,
            max_iter=cfg.lbfgs_max_iter_per_step,
            history_size=cfg.lbfgs_history_size,
            line_search_fn=None,
        )

    hard_label_value = true_label if inferred_label is None else inferred_label
    hard_label = torch.tensor([hard_label_value], dtype=torch.long, device=device)

    if progress_path.exists():
        progress_path.unlink()

    best_match = float("inf")
    best_objective = float("inf")
    best_iteration = 0
    best_image = _image_from_parameter(
        image_parameter.detach(), cfg.image_parameterization
    ).clone()
    best_predicted_label = hard_label_value
    final_match = float("nan")
    final_objective = float("nan")

    latest: dict[str, object] = {}
    _synchronize(device)
    start = time.perf_counter()
    last_print_time = start
    last_print_iteration = 0

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        image_01 = _image_from_parameter(image_parameter, cfg.image_parameterization)
        raw_candidate = compute_named_gradients(
            model=model,
            image_01=image_01,
            label=hard_label if label_logits is None else None,
            parameter_scope=cfg.parameter_scope,
            create_graph=True,
            mean=spec.mean,
            std=spec.std,
            label_logits=label_logits,
        )
        candidate_observation = _candidate_observation(
            raw_candidate,
            cfg,
            settings,
            original_shapes,
        )
        match = _matching_loss(
            candidate_observation,
            target_on_device,
            cfg.matching_loss,
            relative_l2_floor=cfg.relative_l2_floor,
        )
        tv = total_variation(image_01)
        image_l2 = torch.mean((image_01 - 0.5).square())
        objective = match + cfg.tv_weight * tv + cfg.image_l2_weight * image_l2

        if label_logits is not None and cfg.label_entropy_weight > 0:
            probabilities = torch.softmax(label_logits, dim=0)
            entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum()
            objective = objective + cfg.label_entropy_weight * entropy

        gradients = torch.autograd.grad(
            objective,
            attack_parameters,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )
        for parameter, gradient in zip(attack_parameters, gradients):
            parameter.grad = gradient
        if cfg.attack_grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(attack_parameters, cfg.attack_grad_clip_norm)

        predicted_label = (
            hard_label_value
            if label_logits is None
            else int(torch.argmax(label_logits.detach()).item())
        )
        latest.clear()
        latest.update(
            objective=float(objective.detach().item()),
            match=float(match.detach().item()),
            tv=float(tv.detach().item()),
            image_l2=float(image_l2.detach().item()),
            predicted_label=predicted_label,
            image=image_01.detach().clone(),
        )
        return objective

    for iteration in range(1, cfg.iterations + 1):
        if cfg.optimizer == "adam":
            loss = closure()
            del loss
            optimizer.step()
        else:
            optimizer.step(closure)

        if cfg.image_parameterization == "direct":
            with torch.no_grad():
                image_parameter.clamp_(0, 1)

        if not latest:
            raise RuntimeError("Attack optimizer did not evaluate the closure.")

        current_match = float(latest["match"])
        current_objective = float(latest["objective"])
        current_predicted_label = int(latest["predicted_label"])
        current_image = latest["image"]
        assert isinstance(current_image, torch.Tensor)

        if current_match < best_match:
            best_match = current_match
            best_objective = current_objective
            best_iteration = iteration
            best_image = current_image.detach().clone()
            best_predicted_label = current_predicted_label

        final_match = current_match
        final_objective = current_objective

        should_save = iteration == 1 or iteration % cfg.save_every == 0 or iteration == cfg.iterations
        should_print = iteration == 1 or iteration % cfg.print_every == 0 or iteration == cfg.iterations
        if should_save or should_print:
            _synchronize(device)
            now = time.perf_counter()
            elapsed = now - start
            interval_iterations = iteration - last_print_iteration
            interval_seconds = now - last_print_time
            seconds_per_iteration = interval_seconds / max(interval_iterations, 1)
            eta_seconds = seconds_per_iteration * max(cfg.iterations - iteration, 0)
            row = {
                "iteration": iteration,
                "objective": current_objective,
                "match_loss": current_match,
                "tv": float(latest["tv"]),
                "image_l2": float(latest["image_l2"]),
                "predicted_label": current_predicted_label,
                "best_match_loss": best_match,
                "best_iteration": best_iteration,
                "elapsed_seconds": elapsed,
                "seconds_per_iteration": seconds_per_iteration,
                "eta_seconds": eta_seconds,
            }
            if should_save:
                _append_csv(progress_path, row)
            if should_print:
                print(
                    f"  attack={cfg.attack_name:4s} obs={cfg.observation_mode:7s} "
                    f"iter={iteration:5d}/{cfg.iterations} | match={current_match:.6e} | "
                    f"label={current_predicted_label}/{true_label} | "
                    f"{seconds_per_iteration:.3f}s/iter | ETA={eta_seconds / 60:.1f}m",
                    flush=True,
                )
                last_print_time = now
                last_print_iteration = iteration

    final_image = _image_from_parameter(
        image_parameter.detach(), cfg.image_parameterization
    )
    return ClassicAttackResult(
        best_image=best_image,
        final_image=final_image,
        predicted_label=best_predicted_label,
        inferred_label=inferred_label,
        label_source=label_source,
        best_iteration=best_iteration,
        best_objective=best_objective,
        best_match_loss=best_match,
        final_objective=final_objective,
        final_match_loss=final_match,
    )


def _plot_comparison(
    target: torch.Tensor,
    best: torch.Tensor,
    final: torch.Tensor,
    cfg: ClassicAttackConfig,
    sample_index: int,
    output_dir: Path,
) -> None:
    images = [target, best, final]
    labels = ["Target", f"{cfg.attack_name.upper()} best", f"{cfg.attack_name.upper()} final"]
    figure, axes = plt.subplots(1, 3, figsize=(6.3, 2.3), squeeze=False)
    for axis, label, image in zip(axes[0], labels, images):
        array = image.squeeze(0).permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy()
        axis.imshow(array)
        axis.set_title(label, fontsize=9)
        axis.axis("off")
    figure.tight_layout(pad=0.35)
    stem = f"sample_{sample_index:05d}_{cfg.attack_name}_{cfg.observation_mode}_comparison"
    for extension in ("pdf", "svg", "png"):
        kwargs = {"dpi": 600} if extension == "png" else {}
        figure.savefig(output_dir / f"{stem}.{extension}", bbox_inches="tight", **kwargs)
    plt.close(figure)


def _write_summary(metrics_path: Path, output_path: Path) -> None:
    rows = list(csv.DictReader(metrics_path.open("r", encoding="utf-8")))
    summary: dict[str, object] = {"num_samples": len(rows)}
    for metric in ("mse", "psnr", "ssim", "lpips", "attack_seconds"):
        values: list[float] = []
        for row in rows:
            raw = row.get(metric, "")
            if raw in {"", "None", "nan"}:
                continue
            value = float(raw)
            if np.isfinite(value):
                values.append(value)
        summary[f"{metric}_mean"] = None if not values else float(np.mean(values))
        summary[f"{metric}_std"] = None if len(values) < 2 else float(np.std(values, ddof=1))
    if rows:
        summary["label_accuracy"] = float(
            np.mean([int(row["predicted_label"] == row["true_label"]) for row in rows])
        )
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def run_classic_attack_experiment(cfg: ClassicAttackConfig) -> dict[str, object]:
    cfg.validate()
    set_global_seed(cfg.seed, deterministic=True)
    device = resolve_device(cfg.device)
    output_dir = ensure_dir(cfg.output_dir)
    metrics_path = output_dir / "classic_attack_metrics.csv"
    if metrics_path.exists():
        metrics_path.unlink()

    model = _load_model(cfg, device)
    _resolve_attack_parameters(cfg, model)
    cfg.save(output_dir / "resolved_classic_attack_config.json")
    dataset = _build_target_dataset(cfg)
    settings = (
        _make_projection_settings(cfg)
        if cfg.observation_mode in {"decoded", "encoded"}
        else None
    )
    optional_lpips = OptionalLPIPS(device, enabled=cfg.compute_optional_lpips)

    print("=" * 96)
    print(
        f"Classic gradient inversion | attack={cfg.attack_name.upper()} | "
        f"observation={cfg.observation_mode} | device={device}"
    )
    print(
        f"checkpoint={cfg.checkpoint_path} | samples={cfg.sample_indices} | "
        f"iterations={cfg.iterations} | optimizer={cfg.optimizer} | "
        f"loss={cfg.matching_loss} | image_param={cfg.image_parameterization} | "
        f"label_mode={cfg.label_mode}"
    )
    if settings is not None:
        print(
            f"DRPE ratio={cfg.compression_ratio}, dynamic={cfg.dynamic_projection}, "
            f"round={cfg.projection_round}, clip_norm={cfg.observation_clip_norm}, "
            f"noise_std={cfg.noise_std}, noise_seed={cfg.noise_seed}"
        )
    print("=" * 96)

    for sample_index in cfg.sample_indices or []:
        image, true_label = dataset[sample_index]
        target_image = image.unsqueeze(0).to(device)
        target_label_tensor = torch.tensor([true_label], dtype=torch.long, device=device)
        raw_gradients = compute_named_gradients(
            model=model,
            image_01=target_image,
            label=target_label_tensor,
            parameter_scope=cfg.parameter_scope,
            create_graph=False,
            mean=get_dataset_spec(cfg.dataset).mean,
            std=get_dataset_spec(cfg.dataset).std,
        )
        original_shapes = {name: tensor.shape for name, tensor in raw_gradients.items()}
        observation = _target_observation(raw_gradients, cfg, settings)

        sample_dir = ensure_dir(output_dir / f"sample_{sample_index:05d}")
        save_image(target_image.detach().cpu(), sample_dir / "target.png")
        if cfg.save_observations:
            torch.save(observation, sample_dir / f"observation_{cfg.observation_mode}.pt")

        attack_seed = cfg.seed + sample_index * 1009
        original_seed = cfg.seed
        cfg.seed = attack_seed
        start = time.perf_counter()
        result = _run_single_attack(
            model=model,
            target_observation=observation,
            true_label=true_label,
            cfg=cfg,
            settings=settings,
            original_shapes=original_shapes,
            device=device,
            progress_path=sample_dir / f"progress_{cfg.attack_name}_{cfg.observation_mode}.csv",
        )
        cfg.seed = original_seed
        elapsed = time.perf_counter() - start

        best = result.best_image.to(device)
        final = result.final_image.to(device)
        save_image(best.detach().cpu(), sample_dir / f"reconstruction_{cfg.attack_name}_{cfg.observation_mode}_best.png")
        save_image(final.detach().cpu(), sample_dir / f"reconstruction_{cfg.attack_name}_{cfg.observation_mode}_final.png")
        _plot_comparison(target_image.detach().cpu(), best.cpu(), final.cpu(), cfg, sample_index, sample_dir)

        lpips_value = optional_lpips(target_image, best)
        row = {
            "sample_index": sample_index,
            "attack_name": cfg.attack_name,
            "observation_mode": cfg.observation_mode,
            "true_label": true_label,
            "predicted_label": result.predicted_label,
            "inferred_label": "" if result.inferred_label is None else result.inferred_label,
            "label_source": result.label_source,
            "label_correct": int(result.predicted_label == true_label),
            "mse": mse_01(target_image, best),
            "psnr": psnr_01(target_image, best),
            "ssim": ssim_01(target_image, best),
            "lpips": "" if lpips_value is None else lpips_value,
            "best_iteration": result.best_iteration,
            "best_objective": result.best_objective,
            "best_match_loss": result.best_match_loss,
            "final_objective": result.final_objective,
            "final_match_loss": result.final_match_loss,
            "attack_seconds": elapsed,
        }
        _append_csv(metrics_path, row)
        print(
            f"sample={sample_index:05d} attack={cfg.attack_name.upper():4s} "
            f"obs={cfg.observation_mode:7s} | PSNR={float(row['psnr']):6.2f} | "
            f"SSIM={float(row['ssim']):.4f} | label={result.predicted_label}/{true_label} | "
            f"best_iter={result.best_iteration} | {elapsed:.1f}s"
        )

    summary_path = output_dir / "classic_attack_summary.json"
    _write_summary(metrics_path, summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print("-" * 96)
    print(f"Classic attack artifacts saved to: {output_dir.resolve()}")
    return summary
