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

from top_k_baseline.top_k import SparseTensor, encode_top_k_tensor


@dataclass
class TopKConfig:
    # Reproducibility and paths
    seed: int = 42
    partition_seed: int = 42
    data_root: str = "./data"
    output_dir: str = "./outputs/cifar10/resnet18/top_k/commmatched_seed42"
    device: str = "auto"
    num_workers: int = 0
    dataset_download: bool = True

    # Dataset and federated split
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

    # Server/checkpoint
    server_lr: float = 1.0
    early_stop_patience: int = 0
    eval_every: int = 1
    save_every: int = 20

    # Top-k
    keep_ratio: float = 0.15
    selection_scope: str = "tensor"
    error_feedback: bool = False

    # Runtime
    deterministic: bool = True
    pin_memory: bool = True

    def validate(self) -> None:
        self.dataset = self.dataset.lower()
        if self.dataset != "cifar10":
            raise ValueError("This baseline package is currently configured for CIFAR-10.")
        if self.model_name != "resnet18":
            raise ValueError("This baseline package is currently configured for ResNet-18.")
        if self.norm not in {"group", "batch"}:
            raise ValueError("norm must be group/batch.")
        if not (0.0 < self.client_fraction <= 1.0):
            raise ValueError("client_fraction must be in (0, 1].")
        if not (0.0 < self.keep_ratio <= 1.0):
            raise ValueError("keep_ratio must be in (0, 1].")
        if self.selection_scope != "tensor":
            raise ValueError("Only tensor-wise Top-k is implemented.")
        if self.error_feedback:
            raise ValueError(
                "This comparison baseline intentionally uses Top-k without error feedback. "
                "Set error_feedback=false."
            )
        if self.rounds < 1 or self.local_epochs < 1:
            raise ValueError("rounds/local_epochs must be positive.")
        if self.batch_size < 1 or self.eval_batch_size < 1:
            raise ValueError("batch sizes must be positive.")

    def data_config(self) -> ExperimentConfig:
        """
        Build a FedAvg-shaped config only to reuse the project's dataset
        partitioning/data-loader code. This does NOT run FedAvg.
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
    payload: dict[str, SparseTensor]
    floating_buffers: dict[str, torch.Tensor]
    integer_buffers: dict[str, torch.Tensor]
    local_loss: float
    raw_update_norm: float
    selected_count: int
    total_parameter_count: int
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


def load_config(path: str | Path, overrides: list[str]) -> TopKConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    valid = {f.name for f in fields(TopKConfig)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown Top-k config fields: {sorted(unknown)}")
    cfg = TopKConfig(**data)
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


def learning_rate_for_round(cfg: TopKConfig, round_index: int) -> float:
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
    cfg: TopKConfig,
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

            batch_size = inputs.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

    return total_loss / max(total_examples, 1)


def run_client(
    client_id: int,
    dataloader,
    global_model: nn.Module,
    cfg: TopKConfig,
    learning_rate: float,
    device: torch.device,
) -> ClientResult:
    """
    Memory-conscious client:
    - server/global model remains on CPU
    - one local copy is trained on GPU
    - parameter deltas are encoded one tensor at a time
    - no full dense update dictionary is retained
    """
    local_model = copy.deepcopy(global_model).to(device)
    local_loss = train_local_model(
        local_model=local_model,
        dataloader=dataloader,
        cfg=cfg,
        learning_rate=learning_rate,
        device=device,
    )

    global_parameters = dict(global_model.named_parameters())
    payload: dict[str, SparseTensor] = {}
    update_norm_sq = 0.0
    selected_total = 0
    parameter_total = 0
    payload_bytes = 0

    for name, local_parameter in local_model.named_parameters():
        global_parameter = global_parameters[name]
        delta = (
            local_parameter.detach().float().cpu()
            - global_parameter.detach().float().cpu()
        )
        update_norm_sq += float(torch.sum(delta.double().square()).item())

        sparse = encode_top_k_tensor(delta, cfg.keep_ratio)
        payload[name] = sparse
        selected_total += sparse.selected_count
        parameter_total += sparse.numel
        payload_bytes += sparse.payload_bytes
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
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ClientResult(
        client_id=client_id,
        num_samples=num_samples,
        payload=payload,
        floating_buffers=floating_buffers,
        integer_buffers=integer_buffers,
        local_loss=local_loss,
        raw_update_norm=update_norm_sq ** 0.5,
        selected_count=selected_total,
        total_parameter_count=parameter_total,
        uplink_bytes=payload_bytes + buffer_bytes,
    )


@torch.no_grad()
def aggregate_sparse(
    model: nn.Module,
    results: list[ClientResult],
    server_lr: float,
) -> float:
    if not results:
        raise ValueError("Cannot aggregate an empty client result list.")

    total_samples = sum(r.num_samples for r in results)
    decoded_norm_sq = 0.0

    for name, parameter in model.named_parameters():
        accumulator = torch.zeros(parameter.numel(), dtype=torch.float32, device="cpu")

        for result in results:
            sparse = result.payload[name]
            weight = result.num_samples / total_samples
            indices = sparse.indices.to(dtype=torch.long)
            accumulator.index_add_(
                0,
                indices,
                sparse.values.float() * weight,
            )

        decoded_norm_sq += float(torch.sum(accumulator.double().square()).item())
        parameter.add_(
            accumulator.reshape(parameter.shape).to(dtype=parameter.dtype),
            alpha=server_lr,
        )

    # Same buffer aggregation convention as the existing server.
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
def evaluate(model: nn.Module, dataloader, device: torch.device) -> tuple[float, float]:
    """
    Move the server model to the training device only for evaluation, then
    return it to CPU so local-client training starts from a CPU server model.
    """
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


def run_experiment(cfg: TopKConfig) -> dict[str, float]:
    set_global_seed(cfg.seed, deterministic=cfg.deterministic)
    device = resolve_device(cfg.device)
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

    data_cfg = cfg.data_config()
    data = build_federated_data(data_cfg)
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
    selected_clients_per_round = min(
        cfg.num_clients,
        max(1, int(round(cfg.num_clients * cfg.client_fraction))),
    )

    raw_parameter_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )

    best_val_accuracy = float("-inf")
    best_round = 0
    rounds_without_improvement = 0
    cumulative_uplink = 0
    cumulative_downlink = 0

    best_checkpoint = output_dir / "best_model.pt"
    last_checkpoint = output_dir / "last_model.pt"

    print("=" * 104)
    print(
        f"Dataset={cfg.dataset} | Model={cfg.model_name} | "
        f"Method=TOP-K | Device={device}"
    )
    print(
        f"Clients={cfg.num_clients}, selected/round={selected_clients_per_round}, "
        f"alpha={cfg.dirichlet_alpha}, local epochs={cfg.local_epochs}, batch={cfg.batch_size}"
    )
    print(
        f"Keep ratio={cfg.keep_ratio}, scope={cfg.selection_scope}, "
        f"error_feedback={cfg.error_feedback}, value=float32, index=int32, "
        f"expected≈{2.0 * cfg.keep_ratio:.3f}x raw"
    )
    print(f"Server model=CPU during client training | Client training={device}")
    print("=" * 104)

    for round_index in range(1, cfg.rounds + 1):
        start = time.perf_counter()
        lr = learning_rate_for_round(cfg, round_index)

        selected_ids = selection_rng.choice(
            cfg.num_clients,
            size=selected_clients_per_round,
            replace=False,
        ).tolist()

        results = [
            run_client(
                client_id=cid,
                dataloader=data.client_loaders[cid],
                global_model=model,
                cfg=cfg,
                learning_rate=lr,
                device=device,
            )
            for cid in selected_ids
        ]

        decoded_update_norm = aggregate_sparse(
            model=model,
            results=results,
            server_lr=cfg.server_lr,
        )

        total_uplink_bytes = sum(r.uplink_bytes for r in results)
        round_downlink = raw_parameter_bytes * selected_clients_per_round
        cumulative_uplink += total_uplink_bytes
        cumulative_downlink += round_downlink

        selected_fraction = (
            sum(r.selected_count for r in results)
            / max(sum(r.total_parameter_count for r in results), 1)
        )

        val_loss = None
        val_accuracy = None
        if round_index % cfg.eval_every == 0:
            val_loss, val_accuracy = evaluate(model, data.val_loader, device)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_round = round_index
                rounds_without_improvement = 0
                save_checkpoint(model, best_checkpoint)
            else:
                rounds_without_improvement += 1

        elapsed = time.perf_counter() - start
        raw_uplink = raw_parameter_bytes * selected_clients_per_round
        ratio = total_uplink_bytes / max(raw_uplink, 1)

        append_csv(
            {
                "round": round_index,
                "learning_rate": lr,
                "selected_clients": "|".join(map(str, selected_ids)),
                "mean_client_loss": sum(r.local_loss for r in results) / len(results),
                "mean_raw_update_norm": sum(r.raw_update_norm for r in results) / len(results),
                "decoded_update_norm": decoded_update_norm,
                "selected_fraction": selected_fraction,
                "round_uplink_mib": bytes_to_mib(total_uplink_bytes),
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
                f"selected={selected_fraction:.4f} | "
                f"uplink={bytes_to_mib(total_uplink_bytes):.2f} MiB "
                f"({ratio:.3f}x raw) | {elapsed:.1f}s"
            )

        if round_index % cfg.save_every == 0:
            save_checkpoint(model, output_dir / f"checkpoint_round_{round_index:04d}.pt")

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

    save_checkpoint(model, last_checkpoint)

    if best_checkpoint.exists():
        state = torch.load(best_checkpoint, map_location="cpu")
        model.load_state_dict(state)

    test_loss, test_accuracy = evaluate(model, data.test_loader, device)

    summary = {
        "dataset": cfg.dataset,
        "model": cfg.model_name,
        "method": "top_k",
        "selection_scope": cfg.selection_scope,
        "keep_ratio": float(cfg.keep_ratio),
        "error_feedback": bool(cfg.error_feedback),
        "seed": int(cfg.seed),
        "partition_seed": int(cfg.partition_seed),
        "best_round": float(best_round),
        "best_validation_accuracy": float(best_val_accuracy),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "cumulative_uplink_mib": bytes_to_mib(cumulative_uplink),
        "cumulative_downlink_mib": bytes_to_mib(cumulative_downlink),
    }
    save_json(summary, output_dir / "summary.json")

    print("-" * 104)
    print(
        f"Best round={best_round}, best val={best_val_accuracy:.2f}%, "
        f"test={test_accuracy:.2f}%"
    )
    print(f"Artifacts saved to: {output_dir.resolve()}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run independent communication-matched tensor-wise Top-k baseline."
    )
    parser.add_argument("--config", required=True, help="Path to Top-k JSON config.")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    config = load_config(args.config, args.overrides)
    run_experiment(config)
