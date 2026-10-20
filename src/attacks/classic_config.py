from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ClassicAttackConfig:
    seed: int = 42
    dataset: str = "cifar10"
    data_root: str = "./data"
    dataset_download: bool = True
    output_dir: str = "./outputs/attacks"
    device: str = "auto"
    checkpoint_path: Optional[str] = None
    training_summary_path: Optional[str] = None

    model_name: str = "resnet18"
    num_classes: int = 10
    norm: str = "group"
    target_split: str = "train"
    sample_indices: List[int] | None = None

    attack_name: str = "dlg"
    label_mode: str = "optimize"
    observation_mode: str = "raw"
    parameter_scope: str = "all"
    iterations: int = 2000
    attack_lr: float = 0.1
    optimizer: str = "adam"
    matching_loss: str = "cosine"
    tv_weight: float = 1e-4
    image_l2_weight: float = 0.0
    label_entropy_weight: float = 0.001
    init: str = "uniform"
    image_parameterization: str = "sigmoid"
    label_init: str = "zeros"
    label_lr: Optional[float] = 0.05
    attack_grad_clip_norm: Optional[float] = None
    observation_clip_norm: Optional[float] = None
    calibration_path: Optional[str] = None
    noise_eta: float = 2.0
    relative_l2_floor: float = 1e-8
    save_every: int = 25
    print_every: int = 25
    paired_initialization: bool = True
    lbfgs_history_size: int = 100
    lbfgs_max_iter_per_step: int = 1
    save_observations: bool = False
    compute_optional_lpips: bool = True

    compression_ratio: float = 0.3
    block_size: int = 512
    dynamic_projection: bool = True
    projection_seed: int = 2026
    projection_round: Optional[int] = None
    reconstruction_mode: str = "projected"
    projection_device: str = "cpu"
    projection_batch_blocks: int = 256
    noise_std: Optional[float] = None
    noise_seed: int = 314159
    observed_client_id: int = 0

    def __post_init__(self):
        if self.sample_indices is None:
            self.sample_indices = [1]

    def validate(self):
        if self.dataset not in {"cifar10", "cifar100", "tiny_imagenet"}:
            raise ValueError("Unsupported dataset.")
        if self.target_split not in {"train", "test"}:
            raise ValueError("target_split must be train/test.")
        if not self.sample_indices:
            raise ValueError("sample_indices cannot be empty.")
        if self.attack_name not in {"dlg", "idlg"}:
            raise ValueError("attack_name must be dlg/idlg.")
        if self.label_mode not in {"optimize", "known"}:
            raise ValueError("label_mode must be optimize/known.")
        if self.observation_mode not in {"raw", "decoded", "encoded"}:
            raise ValueError("observation_mode must be raw/decoded/encoded.")
        if self.parameter_scope not in {"all", "head_and_last_block", "last_layer"}:
            raise ValueError("Unsupported parameter_scope.")
        if self.optimizer not in {"lbfgs", "adam"}:
            raise ValueError("optimizer must be lbfgs/adam.")
        if self.matching_loss not in {"l2_sum", "l2_mean", "relative_l2", "cosine"}:
            raise ValueError("Unsupported matching_loss.")
        if self.iterations < 1 or self.attack_lr <= 0:
            raise ValueError("Invalid attack budget.")
        if self.block_size < 2 or self.block_size & (self.block_size - 1):
            raise ValueError("block_size must be a power of two.")
        if self.noise_std is not None and self.noise_std < 0:
            raise ValueError("noise_std must be non-negative or null.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_classic_attack_config(path: str | Path) -> ClassicAttackConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    valid = {field.name for field in fields(ClassicAttackConfig)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown classic attack fields: {sorted(unknown)}")
    cfg = ClassicAttackConfig(**data)
    cfg.validate()
    return cfg


def _parse_scalar(text: str) -> Any:
    lower = text.lower()
    if lower in {"true", "false"}: return lower == "true"
    if lower in {"none", "null"}: return None
    if text.startswith("["): return json.loads(text)
    try:
        return float(text) if any(c in text for c in ".eE") else int(text)
    except ValueError:
        return text


def apply_classic_attack_overrides(cfg: ClassicAttackConfig, overrides: list[str]):
    valid = {field.name for field in fields(ClassicAttackConfig)}
    for item in overrides:
        if "=" not in item: raise ValueError(f"Override must be key=value: {item}")
        key, raw = item.split("=", 1)
        if key not in valid: raise ValueError(f"Unknown field: {key}")
        setattr(cfg, key, _parse_scalar(raw.strip()))
    cfg.__post_init__()
    cfg.validate()
    return cfg
