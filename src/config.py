from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import torch.nn as nn

from .projection import encoded_numel


SUPPORTED_DATASETS = {"cifar10", "cifar100", "tiny_imagenet"}
SUPPORTED_MODELS = {"resnet18", "mobilenet_v2"}


@dataclass
class ExperimentConfig:
    # Reproducibility and paths
    seed: int = 42
    partition_seed: int = 42
    data_root: str = "./data"
    output_dir: str = "./outputs/cifar10/resnet18/drpe/seed42"
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

    # Server and checkpoints
    server_lr: float = 1.0
    early_stop_patience: int = 0
    eval_every: int = 1
    save_every: int = 20

    # Method
    method: str = "drpe"  # fedavg / drpe

    # DRPE projection
    compression_ratio: float = 0.3
    block_size: int = 512
    dynamic_projection: bool = True
    projection_seed: int = 2026
    reconstruction_mode: str = "projected"
    projection_device: str = "cpu"

    # Complete DRPE = P75 clipping + encoded Gaussian perturbation.
    privacy_enabled: bool = True
    calibration_path: Optional[str] = None
    clip_norm: Optional[float] = None
    noise_eta: float = 2.0
    noise_std: Optional[float] = None
    noise_seed: int = 314159

    # Runtime
    deterministic: bool = True
    pin_memory: bool = True

    def validate(self) -> None:
        self.dataset = self.dataset.lower()
        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        if self.model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model_name: {self.model_name}")
        if self.method not in {"drpe", "fedavg"}:
            raise ValueError("method must be 'drpe' or 'fedavg'.")
        if not (0 < self.client_fraction <= 1):
            raise ValueError("client_fraction must be in (0, 1].")
        if self.num_clients < 2:
            raise ValueError("num_clients must be at least 2.")
        if self.local_epochs < 1 or self.rounds < 1:
            raise ValueError("local_epochs and rounds must be positive.")
        if not (0 < self.compression_ratio <= 1):
            raise ValueError("compression_ratio must be in (0, 1].")
        if self.block_size < 2 or self.block_size & (self.block_size - 1):
            raise ValueError("block_size must be a power of two.")
        if self.reconstruction_mode not in {"projected", "unbiased"}:
            raise ValueError("Unsupported reconstruction_mode.")
        if self.norm not in {"group", "batch"}:
            raise ValueError("norm must be 'group' or 'batch'.")
        if self.noise_eta < 0:
            raise ValueError("noise_eta must be non-negative.")
        if self.noise_std is not None and self.noise_std < 0:
            raise ValueError("noise_std must be non-negative or null.")
        if self.clip_norm is not None and self.clip_norm <= 0:
            raise ValueError("clip_norm must be positive or null.")
        if not (0 <= self.validation_ratio < 0.5):
            raise ValueError("validation_ratio must be in [0, 0.5).")
        if self.eval_batch_size < 1 or self.batch_size < 1:
            raise ValueError("batch sizes must be positive.")
        if self.method == "fedavg":
            self.privacy_enabled = False
            self.clip_norm = None
            self.noise_std = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def model_encoded_numel(model: nn.Module, block_size: int, compression_ratio: float) -> int:
    return sum(
        encoded_numel(parameter.numel(), block_size, compression_ratio)
        for parameter in model.parameters()
    )


def resolve_drpe_privacy_parameters(cfg: ExperimentConfig, model: nn.Module) -> Dict[str, float | int | str | None]:
    """Resolve P75 and eta-based sigma before clients are created."""
    if cfg.method != "drpe" or not cfg.privacy_enabled:
        cfg.clip_norm = None
        cfg.noise_std = 0.0
        return {
            "privacy_enabled": False,
            "clip_norm": None,
            "noise_eta": 0.0,
            "noise_std": 0.0,
            "encoded_parameter_count": model_encoded_numel(
                model, cfg.block_size, cfg.compression_ratio
            ) if cfg.method == "drpe" else sum(p.numel() for p in model.parameters()),
        }

    calibration: Dict[str, Any] = {}
    if cfg.calibration_path:
        path = Path(cfg.calibration_path)
        if not path.exists():
            raise FileNotFoundError(
                f"DRPE calibration file not found: {path}. "
                "Run scripts/run_calibration.py with the matching dataset/model config first."
            )
        calibration = json.loads(path.read_text(encoding="utf-8"))

    if cfg.clip_norm is None:
        candidate = calibration.get("selected_clip_norm", calibration.get("p75"))
        if candidate is None:
            raise ValueError("Complete DRPE requires clip_norm or a calibration file with selected_clip_norm/p75.")
        cfg.clip_norm = float(candidate)

    encoded_count = model_encoded_numel(model, cfg.block_size, cfg.compression_ratio)
    if cfg.noise_std is None:
        cfg.noise_std = float(cfg.noise_eta * cfg.clip_norm / math.sqrt(encoded_count))

    return {
        "privacy_enabled": True,
        "clip_norm": float(cfg.clip_norm),
        "noise_eta": float(cfg.noise_eta),
        "noise_std": float(cfg.noise_std),
        "encoded_parameter_count": int(encoded_count),
        "calibration_path": cfg.calibration_path,
    }


def load_config(path: str | Path) -> ExperimentConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    valid_names = {field.name for field in fields(ExperimentConfig)}
    unknown = set(data) - valid_names
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    cfg = ExperimentConfig(**data)
    cfg.validate()
    return cfg


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


def apply_overrides(cfg: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    valid_names = {field.name for field in fields(ExperimentConfig)}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if key not in valid_names:
            raise ValueError(f"Unknown override field: {key}")
        setattr(cfg, key, _parse_scalar(raw_value.strip()))
    cfg.validate()
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run formal FedAvg/DRPE experiments.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    return parser
