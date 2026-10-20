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

from flocora_baseline.flocora import (
    count_flocora_parameters,
    inject_flocora,
    trainable_named_parameters,
    trainable_state_dict,
)


@dataclass
class FLoCoRAConfig:
    # Reproducibility and paths
    seed: int = 42
    partition_seed: int = 42
    data_root: str = "./data"
    output_dir: str = "./outputs/cifar10/resnet18/flocora/commmatched_seed42"
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

    # FLoCoRA
    rank: int = 93
    lora_alpha: float = 1488.0  # 16 * rank, following the paper's stronger CNN scaling
    target_uplink_ratio: float = 0.30

    # Explicitly disabled extras
    quantization_bits: int | None = None
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
        if self.norm != "group":
            raise ValueError("Formal FLoCoRA comparison is configured for GroupNorm.")
        if not (0.0 < self.client_fraction <= 1.0):
            raise ValueError("client_fraction must be in (0, 1].")
        if self.rank < 1 or self.lora_alpha <= 0:
            raise ValueError("rank and lora_alpha must be positive.")
        if self.quantization_bits is not None:
            raise ValueError("Main FLoCoRA baseline is FP32 only; quantization_bits must be null.")
        if self.clip_norm is not None:
            raise ValueError("FLoCoRA comparison baseline must use clip_norm=null.")
        if float(self.noise_std) != 0.0:
            raise ValueError("FLoCoRA comparison baseline must use noise_std=0.")
        if self.rounds < 1 or self.local_epochs < 1:
            raise ValueError("rounds/local_epochs must be positive.")

    def data_config(self) -> ExperimentConfig:
        # Reuse the formal project's partitioning and DataLoader logic only.
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
    trainable_state: dict[str, torch.Tensor]
    local_loss: float
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


def load_config(path: str | Path, overrides: list[str]) -> FLoCoRAConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    valid = {field.name for field in fields(FLoCoRAConfig)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown FLoCoRA config fields: {sorted(unknown)}")

    cfg = FLoCoRAConfig(**data)
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


def learning_rate_for_round(cfg: FLoCoRAConfig, round_index: int) -> float:
    return cfg.learning_rate * (
        cfg.lr_decay_gamma ** ((round_index - 1) // cfg.lr_decay_step)
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
    cfg: FLoCoRAConfig,
    learning_rate: float,
    device: torch.device,
) -> float:
    local_model.train()
    criterion = nn.CrossEntropyLoss()
    trainable = [parameter for parameter in local_model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(
        trainable,
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
    cfg: FLoCoRAConfig,
    learning_rate: float,
    train_device: torch.device,
) -> ClientResult:
    local_model = copy.deepcopy(global_model).to(train_device)
    local_loss = train_local_model(
        local_model=local_model,
        dataloader=dataloader,
        cfg=cfg,
        learning_rate=learning_rate,
        device=train_device,
    )

    local_model.to("cpu")
    if train_device.type == "cuda":
        torch.cuda.empty_cache()

    state = trainable_state_dict(local_model)
    uplink_bytes = sum(t.numel() * t.element_size() for t in state.values())
    num_samples = len(dataloader.dataset)
    del local_model

    return ClientResult(
        client_id=client_id,
        num_samples=num_samples,
        trainable_state=state,
        local_loss=local_loss,
        uplink_bytes=uplink_bytes,
    )


@torch.no_grad()
def aggregate_trainable_states(
    model: nn.Module,
    results: list[ClientResult],
    server_lr: float,
) -> float:
    if not results:
        raise ValueError("Cannot aggregate empty client results.")

    total_samples = sum(result.num_samples for result in results)
    parameter_map = dict(model.named_parameters())
    aggregate_delta_norm_sq = 0.0

    for name, global_parameter in trainable_named_parameters(model):
        averaged = torch.zeros_like(global_parameter.detach().float().cpu())
        for result in results:
            weight = result.num_samples / total_samples
            averaged.add_(result.trainable_state[name], alpha=weight)

        old = global_parameter.detach().float().cpu()
        delta = averaged - old
        aggregate_delta_norm_sq += float(torch.sum(delta.double().square()).item())

        if server_lr == 1.0:
            parameter_map[name].copy_(averaged.to(dtype=global_parameter.dtype))
        else:
            parameter_map[name].add_(
                delta.to(dtype=global_parameter.dtype),
                alpha=server_lr,
            )

    return aggregate_delta_norm_sq ** 0.5


@torch.no_grad()
def evaluate(model: nn.Module, dataloader, device: torch.device) -> tuple[float, float]:
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


def run_experiment(cfg: FLoCoRAConfig) -> dict[str, object]:
    set_global_seed(cfg.seed, deterministic=cfg.deterministic)
    train_device = resolve_device(cfg.device)
    output_dir = ensure_dir(cfg.output_dir)
    metrics_path = output_dir / "metrics.csv"
    if metrics_path.exists():
        metrics_path.unlink()

    spec = get_dataset_spec(cfg.dataset)
    dense_reference = build_model(
        cfg.model_name,
        cfg.num_classes,
        cfg.norm,
        spec.image_size,
    ).cpu()
    raw_parameter_count = sum(p.numel() for p in dense_reference.parameters())
    raw_parameter_bytes = sum(p.numel() * p.element_size() for p in dense_reference.parameters())

    # Reuse exactly the same dense initialization, then inject zero-output LoRA adapters.
    model = dense_reference
    inject_flocora(model, rank=cfg.rank, alpha=cfg.lora_alpha)
    stats = count_flocora_parameters(model, raw_parameter_count)

    trainable_payload_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
        if p.requires_grad
    )

    save_json(asdict(cfg), output_dir / "resolved_config.json")
    save_json(environment(), output_dir / "environment.json")
    save_json(asdict(stats), output_dir / "flocora_parameter_stats.json")

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

    best_val_accuracy = float("-inf")
    best_round = 0
    rounds_without_improvement = 0
    cumulative_uplink = 0
    cumulative_downlink_payload = 0
    cumulative_downlink_full_reference = 0

    best_checkpoint = output_dir / "best_model.pt"
    last_checkpoint = output_dir / "last_model.pt"

    print("=" * 116)
    print(
        f"Dataset={cfg.dataset} | Model={cfg.model_name} | Method=FLoCoRA-FP32 | "
        f"Train device={train_device}"
    )
    print(
        f"Clients={cfg.num_clients}, selected/round={selected_count}, alpha_data={cfg.dirichlet_alpha}, "
        f"local epochs={cfg.local_epochs}, batch={cfg.batch_size}"
    )
    print(
        f"LoRA rank={cfg.rank}, lora_alpha={cfg.lora_alpha:g}, scaling={cfg.lora_alpha/cfg.rank:.3f} | "
        f"Conv LoRA + trainable GroupNorm + trainable final FC"
    )
    print(
        f"Trainable payload={stats.trainable_parameters:,}/{stats.raw_dense_parameters:,} "
        f"({stats.trainable_ratio_vs_dense:.6f}x dense) | adapted convs={stats.conv_layers_adapted}"
    )
    print(
        f"Per-round uplink ({selected_count} clients)={bytes_to_mib(trainable_payload_bytes * selected_count):.2f} MiB | "
        f"FedAvg raw reference={bytes_to_mib(raw_parameter_bytes * selected_count):.2f} MiB"
    )
    print("Quantization=None | clipping=None | noise=0.0")
    print("=" * 116)

    for round_index in range(1, cfg.rounds + 1):
        start = time.perf_counter()
        lr = learning_rate_for_round(cfg, round_index)

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
                train_device=train_device,
            )
            for cid in selected_ids
        ]

        aggregated_update_norm = aggregate_trainable_states(
            model=model,
            results=results,
            server_lr=cfg.server_lr,
        )

        round_uplink = sum(result.uplink_bytes for result in results)
        # FLoCoRA actual repeated server payload: trainable adapter/norm/head state only.
        round_downlink_payload = trainable_payload_bytes * selected_count
        # Keep a FedAvg-style full dense reference for the paper's apples-to-apples tables.
        round_downlink_full_reference = raw_parameter_bytes * selected_count

        cumulative_uplink += round_uplink
        cumulative_downlink_payload += round_downlink_payload
        cumulative_downlink_full_reference += round_downlink_full_reference

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
                "mean_client_loss": sum(result.local_loss for result in results) / len(results),
                "aggregated_trainable_update_norm": aggregated_update_norm,
                "round_uplink_mib": bytes_to_mib(round_uplink),
                "uplink_ratio_vs_raw_parameters": ratio,
                "round_downlink_payload_mib": bytes_to_mib(round_downlink_payload),
                "round_downlink_full_model_reference_mib": bytes_to_mib(round_downlink_full_reference),
                "cumulative_uplink_mib": bytes_to_mib(cumulative_uplink),
                "cumulative_downlink_payload_mib": bytes_to_mib(cumulative_downlink_payload),
                "cumulative_downlink_full_model_reference_mib": bytes_to_mib(cumulative_downlink_full_reference),
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
                f"loss={sum(result.local_loss for result in results)/len(results):.4f} | "
                f"uplink={bytes_to_mib(round_uplink):.2f} MiB ({ratio:.3f}x raw) | {elapsed:.1f}s"
            )

        if round_index % cfg.save_every == 0:
            save_checkpoint(model, output_dir / f"checkpoint_round_{round_index:04d}.pt")

        if (
            val_accuracy is not None
            and cfg.early_stop_patience > 0
            and rounds_without_improvement >= cfg.early_stop_patience
        ):
            print(
                f"Early stop at round {round_index}; best={best_val_accuracy:.2f}% at {best_round}."
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
        "method": "flocora_fp32",
        "rank": int(cfg.rank),
        "lora_alpha": float(cfg.lora_alpha),
        "lora_scaling": float(cfg.lora_alpha / cfg.rank),
        "raw_dense_parameters": int(stats.raw_dense_parameters),
        "trainable_parameters": int(stats.trainable_parameters),
        "trainable_ratio_vs_dense": float(stats.trainable_ratio_vs_dense),
        "adapter_parameters": int(stats.adapter_parameters),
        "norm_parameters": int(stats.norm_parameters),
        "head_parameters": int(stats.head_parameters),
        "conv_layers_adapted": int(stats.conv_layers_adapted),
        "seed": int(cfg.seed),
        "partition_seed": int(cfg.partition_seed),
        "best_round": int(best_round),
        "best_validation_accuracy": float(best_val_accuracy),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "cumulative_uplink_mib": bytes_to_mib(cumulative_uplink),
        "cumulative_downlink_payload_mib": bytes_to_mib(cumulative_downlink_payload),
        "cumulative_downlink_full_model_reference_mib": bytes_to_mib(cumulative_downlink_full_reference),
    }
    save_json(summary, output_dir / "summary.json")

    print("-" * 116)
    print(
        f"Best round={best_round}, best val={best_val_accuracy:.2f}%, test={test_accuracy:.2f}%"
    )
    print(f"Artifacts saved to: {output_dir.resolve()}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run communication-matched FLoCoRA FP32 baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    cfg = load_config(args.config, args.overrides)
    run_experiment(cfg)
