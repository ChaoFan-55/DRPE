from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig
from src.data import build_federated_data, get_dataset_spec
from src.models import build_model
from src.utils import append_csv, bytes_to_mib, ensure_dir, save_json, set_global_seed

from powersgd_baseline.powersgd import (
    PowerSGDSettings,
    clear_powersgd_cache,
    decode_tensor,
    encode_tensor,
    payload_bytes,
    payload_numel_for_shape,
)


@dataclass
class PowerSGDConfig:
    # Reproducibility and paths
    seed: int = 42
    partition_seed: int = 42
    data_root: str = "./data"
    output_dir: str = "./outputs/cifar10/resnet18/powersgd/commmatched_seed42"
    device: str = "auto"
    num_workers: int = 0
    dataset_download: bool = True

    # Dataset / FL split
    dataset: str = "cifar10"
    num_classes: int = 10
    validation_ratio: float = 0.1
    num_clients: int = 10
    client_fraction: float = 0.2
    dirichlet_alpha: float = 0.5
    min_samples_per_client: int = 100
    eval_batch_size: int = 256

    # Model
    model_name: str = "resnet18"
    norm: str = "group"

    # Local optimization
    rounds: int = 200
    local_epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    lr_decay_step: int = 60
    lr_decay_gamma: float = 0.5

    # Server / checkpoint
    server_lr: float = 1.0
    early_stop_patience: int = 0
    eval_every: int = 1
    save_every: int = 20

    # PowerSGD
    rank: int = 97
    power_iterations: int = 1
    powersgd_seed: int = 2026
    compression_device: str = "cuda"
    fallback_dense_if_not_smaller: bool = True
    error_feedback: bool = False

    # Explicitly disabled privacy extras
    clip_norm: float | None = None
    noise_std: float = 0.0

    # Runtime
    deterministic: bool = True
    pin_memory: bool = True

    def validate(self) -> None:
        self.dataset = self.dataset.lower()
        if self.dataset != "cifar10":
            raise ValueError("This package is configured for CIFAR-10.")
        if self.model_name != "resnet18":
            raise ValueError("This package is configured for ResNet-18.")
        if self.norm not in {"group", "batch"}:
            raise ValueError("norm must be group/batch.")
        if not (0.0 < self.client_fraction <= 1.0):
            raise ValueError("client_fraction must be in (0, 1].")
        if self.rank < 1:
            raise ValueError("rank must be positive.")
        if self.power_iterations < 1:
            raise ValueError("power_iterations must be at least 1.")
        if self.error_feedback:
            raise ValueError(
                "This communication-matched baseline intentionally uses "
                "error_feedback=false so the round-20 transmitted observation "
                "is exactly reproducible for the unified inversion protocol."
            )
        if self.clip_norm is not None:
            raise ValueError("PowerSGD comparison baseline must use clip_norm=null.")
        if float(self.noise_std) != 0.0:
            raise ValueError("PowerSGD comparison baseline must use noise_std=0.")
        if self.rounds < 1 or self.local_epochs < 1:
            raise ValueError("rounds/local_epochs must be positive.")

    def data_config(self) -> ExperimentConfig:
        """
        Reuse only the project's dataset partition/DataLoader construction.
        The PowerSGD aggregation/compression loop stays isolated in this folder.
        """
        cfg = ExperimentConfig(
            seed=self.seed,
            partition_seed=self.partition_seed,
            data_root=self.data_root,
            output_dir=self.output_dir,
            device=self.device,
            num_workers=self.num_workers,
            dataset_download=self.dataset_download,
            dataset=self.dataset,
            num_classes=self.num_classes,
            validation_ratio=self.validation_ratio,
            num_clients=self.num_clients,
            client_fraction=self.client_fraction,
            dirichlet_alpha=self.dirichlet_alpha,
            min_samples_per_client=self.min_samples_per_client,
            eval_batch_size=self.eval_batch_size,
            model_name=self.model_name,
            norm=self.norm,
            rounds=self.rounds,
            local_epochs=self.local_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            lr_decay_step=self.lr_decay_step,
            lr_decay_gamma=self.lr_decay_gamma,
            server_lr=self.server_lr,
            early_stop_patience=self.early_stop_patience,
            eval_every=self.eval_every,
            save_every=self.save_every,
            method="fedavg",
            compression_ratio=1.0,
            dynamic_projection=False,
            privacy_enabled=False,
            clip_norm=None,
            noise_eta=0.0,
            noise_std=0.0,
            deterministic=self.deterministic,
            pin_memory=self.pin_memory,
        )
        cfg.validate()
        return cfg


@dataclass
class ClientResult:
    client_id: int
    num_samples: int
    payload: dict[str, dict[str, Any]]
    floating_buffers: dict[str, torch.Tensor]
    integer_buffers: dict[str, torch.Tensor]
    local_loss: float
    raw_update_norm: float
    uplink_bytes: int


def _parse_scalar(text: str) -> Any:
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None
    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text


def load_config(path: str | Path, overrides: list[str]) -> PowerSGDConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    valid = {f.name for f in fields(PowerSGDConfig)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown PowerSGD config fields: {sorted(unknown)}")

    cfg = PowerSGDConfig(**data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in valid:
            raise ValueError(f"Unknown override field: {key}")
        setattr(cfg, key, _parse_scalar(raw.strip()))

    cfg.validate()
    return cfg


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device(requested)


def learning_rate_for_round(cfg: PowerSGDConfig, round_index: int) -> float:
    return cfg.learning_rate * (
        cfg.lr_decay_gamma ** ((round_index - 1) // cfg.lr_decay_step)
    )


def compression_settings(
    cfg: PowerSGDConfig,
    round_index: int,
    client_id: int,
) -> PowerSGDSettings:
    return PowerSGDSettings(
        rank=cfg.rank,
        base_seed=cfg.powersgd_seed,
        round_index=round_index,
        client_id=client_id,
        power_iterations=cfg.power_iterations,
        device=cfg.compression_device,
        fallback_dense_if_not_smaller=cfg.fallback_dense_if_not_smaller,
    )


def environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def train_local_model(
    local_model: nn.Module,
    dataloader,
    cfg: PowerSGDConfig,
    learning_rate: float,
    device: torch.device,
) -> float:
    local_model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        local_model.parameters(),
        lr=learning_rate,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    total_loss = 0.0
    total_examples = 0

    for _ in range(cfg.local_epochs):
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = local_model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch = inputs.size(0)
            total_loss += float(loss.item()) * batch
            total_examples += batch

    return total_loss / max(total_examples, 1)


def run_client(
    client_id: int,
    dataloader,
    global_model: nn.Module,
    cfg: PowerSGDConfig,
    learning_rate: float,
    round_index: int,
    train_device: torch.device,
) -> ClientResult:
    local_model = copy.deepcopy(global_model).to(train_device)

    local_loss = train_local_model(
        local_model,
        dataloader,
        cfg,
        learning_rate,
        train_device,
    )

    # Free most GPU model memory before running low-rank factorization.
    local_model.to("cpu")
    if train_device.type == "cuda":
        torch.cuda.empty_cache()

    global_parameters = dict(global_model.named_parameters())
    payload: dict[str, dict[str, Any]] = {}
    raw_norm_sq = 0.0
    total_payload_bytes = 0

    settings = compression_settings(cfg, round_index, client_id)

    for name, local_parameter in local_model.named_parameters():
        global_parameter = global_parameters[name]
        delta = (
            local_parameter.detach().float()
            - global_parameter.detach().float()
        )
        raw_norm_sq += float(torch.sum(delta.double().square()).item())

        compressed = encode_tensor(
            tensor=delta,
            tensor_name=name,
            settings=settings,
        )
        payload[name] = compressed
        total_payload_bytes += payload_bytes(compressed)
        del delta

    local_buffers = {
        name: buffer.detach().clone().cpu()
        for name, buffer in local_model.named_buffers()
    }
    floating_buffers = {
        name: tensor.float()
        for name, tensor in local_buffers.items()
        if torch.is_floating_point(tensor)
    }
    integer_buffers = {
        name: tensor
        for name, tensor in local_buffers.items()
        if not torch.is_floating_point(tensor)
    }
    buffer_bytes = sum(t.numel() * t.element_size() for t in local_buffers.values())

    num_samples = len(dataloader.dataset)
    del local_model

    return ClientResult(
        client_id=client_id,
        num_samples=num_samples,
        payload=payload,
        floating_buffers=floating_buffers,
        integer_buffers=integer_buffers,
        local_loss=local_loss,
        raw_update_norm=raw_norm_sq ** 0.5,
        uplink_bytes=total_payload_bytes + buffer_bytes,
    )


@torch.no_grad()
def aggregate_payloads(
    model: nn.Module,
    results: list[ClientResult],
    server_lr: float,
) -> float:
    if not results:
        raise ValueError("Cannot aggregate empty results.")

    total_samples = sum(r.num_samples for r in results)
    decoded_norm_sq = 0.0

    for name, parameter in model.named_parameters():
        accumulator = torch.zeros_like(parameter.detach().float().cpu())

        for result in results:
            weight = result.num_samples / total_samples
            decoded = decode_tensor(result.payload[name])
            accumulator.add_(decoded, alpha=weight)

        decoded_norm_sq += float(torch.sum(accumulator.double().square()).item())
        parameter.add_(
            accumulator.to(dtype=parameter.dtype),
            alpha=server_lr,
        )

    buffer_map = dict(model.named_buffers())

    if results[0].floating_buffers:
        for name in results[0].floating_buffers:
            reference = results[0].floating_buffers[name]
            accumulator = torch.zeros_like(reference, dtype=torch.float32)
            for result in results:
                weight = result.num_samples / total_samples
                accumulator.add_(result.floating_buffers[name].float(), alpha=weight)
            if name in buffer_map:
                buffer_map[name].copy_(accumulator.to(dtype=buffer_map[name].dtype))

    if results[0].integer_buffers:
        for name in results[0].integer_buffers:
            stacked = torch.stack([r.integer_buffers[name].long() for r in results])
            value = torch.max(stacked, dim=0).values
            if name in buffer_map:
                buffer_map[name].copy_(value.to(dtype=buffer_map[name].dtype))

    return decoded_norm_sq ** 0.5


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> tuple[float, float]:
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(inputs)
        loss = criterion(logits, targets)

        total_loss += float(loss.item()) * inputs.size(0)
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_examples += inputs.size(0)

    loss_value = total_loss / max(total_examples, 1)
    accuracy = 100.0 * total_correct / max(total_examples, 1)

    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return loss_value, accuracy


def save_checkpoint(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def run_experiment(cfg: PowerSGDConfig) -> dict[str, object]:
    set_global_seed(cfg.seed, deterministic=cfg.deterministic)
    train_device = resolve_device(cfg.device)
    output_dir = ensure_dir(cfg.output_dir)
    metrics_path = output_dir / "metrics.csv"
    if metrics_path.exists():
        metrics_path.unlink()

    spec = get_dataset_spec(cfg.dataset)
    model = build_model(
        cfg.model_name,
        cfg.num_classes,
        cfg.norm,
        spec.image_size,
    ).cpu()

    save_json(asdict(cfg), output_dir / "resolved_config.json")
    save_json(environment(), output_dir / "environment.json")

    data = build_federated_data(cfg.data_config())
    save_json(
        {
            "client_sizes": data.client_sizes,
            "min_client_size": min(data.client_sizes),
            "max_client_size": max(data.client_sizes),
            "mean_client_size": float(np.mean(data.client_sizes)),
            "partition_seed": cfg.partition_seed,
        },
        output_dir / "partition_summary.json",
    )

    selection_rng = np.random.default_rng(cfg.seed + 999)
    selected_count = min(
        cfg.num_clients,
        max(1, int(round(cfg.num_clients * cfg.client_fraction))),
    )

    raw_parameter_count = sum(p.numel() for p in model.parameters())
    raw_parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    predicted_payload_count = sum(
        payload_numel_for_shape(
            p.shape,
            rank=cfg.rank,
            fallback_dense_if_not_smaller=cfg.fallback_dense_if_not_smaller,
        )
        for p in model.parameters()
    )
    predicted_ratio = predicted_payload_count / max(raw_parameter_count, 1)

    best_val_accuracy = float("-inf")
    best_round = 0
    rounds_without_improvement = 0
    cumulative_uplink = 0
    cumulative_downlink = 0

    best_checkpoint = output_dir / "best_model.pt"
    last_checkpoint = output_dir / "last_model.pt"

    print("=" * 108)
    print(
        f"Dataset={cfg.dataset} | Model={cfg.model_name} | "
        f"Method=POWERSGD | Train device={train_device}"
    )
    print(
        f"Clients={cfg.num_clients}, selected/round={selected_count}, "
        f"alpha={cfg.dirichlet_alpha}, local epochs={cfg.local_epochs}, batch={cfg.batch_size}"
    )
    print(
        f"Rank={cfg.rank}, power_iterations={cfg.power_iterations}, "
        f"powersgd_seed={cfg.powersgd_seed}, compression_device={cfg.compression_device}"
    )
    print(
        f"Fallback dense if factors are not smaller={cfg.fallback_dense_if_not_smaller} | "
        f"error_feedback={cfg.error_feedback}"
    )
    print(
        f"Predicted payload={predicted_payload_count:,}/{raw_parameter_count:,} "
        f"({predicted_ratio:.6f}x)"
    )
    print("Privacy extras: clipping=None, noise=0.0")
    print("=" * 108)

    for round_index in range(1, cfg.rounds + 1):
        start = time.perf_counter()
        lr = learning_rate_for_round(cfg, round_index)

        # Cache is only useful inside the current round. Each client has its
        # own deterministic Q0 because client_id participates in the seed.
        clear_powersgd_cache()

        selected_ids = selection_rng.choice(
            cfg.num_clients,
            size=selected_count,
            replace=False,
        ).tolist()

        results = [
            run_client(
                client_id=cid,
                dataloader=data.client_loaders[cid],
                global_model=model,
                cfg=cfg,
                learning_rate=lr,
                round_index=round_index,
                train_device=train_device,
            )
            for cid in selected_ids
        ]

        decoded_update_norm = aggregate_payloads(
            model=model,
            results=results,
            server_lr=cfg.server_lr,
        )

        round_uplink = sum(r.uplink_bytes for r in results)
        round_downlink = raw_parameter_bytes * selected_count
        cumulative_uplink += round_uplink
        cumulative_downlink += round_downlink

        val_loss = None
        val_accuracy = None
        if round_index % cfg.eval_every == 0:
            val_loss, val_accuracy = evaluate(model, data.val_loader, train_device)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_round = round_index
                rounds_without_improvement = 0
                save_checkpoint(model, best_checkpoint)
            else:
                rounds_without_improvement += 1

        elapsed = time.perf_counter() - start
        raw_uplink = raw_parameter_bytes * selected_count
        ratio = round_uplink / max(raw_uplink, 1)

        append_csv(
            {
                "round": round_index,
                "learning_rate": lr,
                "selected_clients": "|".join(map(str, selected_ids)),
                "mean_client_loss": sum(r.local_loss for r in results) / len(results),
                "mean_raw_update_norm": sum(r.raw_update_norm for r in results) / len(results),
                "decoded_update_norm": decoded_update_norm,
                "round_uplink_mib": bytes_to_mib(round_uplink),
                "round_downlink_mib": bytes_to_mib(round_downlink),
                "uplink_ratio_vs_raw_parameters": ratio,
                "cumulative_uplink_mib": bytes_to_mib(cumulative_uplink),
                "cumulative_downlink_mib": bytes_to_mib(cumulative_downlink),
                "val_loss": "" if val_loss is None else val_loss,
                "val_accuracy": "" if val_accuracy is None else val_accuracy,
                "round_seconds": elapsed,
            },
            metrics_path,
        )

        if round_index == 1 or round_index % 5 == 0:
            val_text = "not evaluated" if val_accuracy is None else f"{val_accuracy:.2f}%"
            print(
                f"Round {round_index:03d} | lr={lr:.5f} | val={val_text} | "
                f"loss={sum(r.local_loss for r in results) / len(results):.4f} | "
                f"uplink={bytes_to_mib(round_uplink):.2f} MiB ({ratio:.3f}x raw) | "
                f"{elapsed:.1f}s"
            )

        if round_index % cfg.save_every == 0:
            save_checkpoint(
                model,
                output_dir / f"checkpoint_round_{round_index:04d}.pt",
            )

        if (
            val_accuracy is not None
            and cfg.early_stop_patience > 0
            and rounds_without_improvement >= cfg.early_stop_patience
        ):
            print(
                f"Early stop at round {round_index}; "
                f"best={best_val_accuracy:.2f}% at {best_round}."
            )
            break

        del results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_checkpoint(model, last_checkpoint)

    if best_checkpoint.exists():
        state = torch.load(best_checkpoint, map_location="cpu")
        model.load_state_dict(state)

    test_loss, test_accuracy = evaluate(model, data.test_loader, train_device)

    summary = {
        "dataset": cfg.dataset,
        "model": cfg.model_name,
        "method": "powersgd",
        "rank": int(cfg.rank),
        "power_iterations": int(cfg.power_iterations),
        "powersgd_seed": int(cfg.powersgd_seed),
        "error_feedback": bool(cfg.error_feedback),
        "fallback_dense_if_not_smaller": bool(cfg.fallback_dense_if_not_smaller),
        "predicted_payload_count": int(predicted_payload_count),
        "raw_parameter_count": int(raw_parameter_count),
        "predicted_payload_ratio": float(predicted_ratio),
        "seed": int(cfg.seed),
        "partition_seed": int(cfg.partition_seed),
        "best_round": int(best_round),
        "best_validation_accuracy": float(best_val_accuracy),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "cumulative_uplink_mib": bytes_to_mib(cumulative_uplink),
        "cumulative_downlink_mib": bytes_to_mib(cumulative_downlink),
    }
    save_json(summary, output_dir / "summary.json")

    print("-" * 108)
    print(
        f"Best round={best_round}, best val={best_val_accuracy:.2f}%, "
        f"test={test_accuracy:.2f}%"
    )
    print(f"Artifacts saved to: {output_dir.resolve()}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run communication-matched PowerSGD baseline."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    cfg = load_config(args.config, args.overrides)
    run_experiment(cfg)
