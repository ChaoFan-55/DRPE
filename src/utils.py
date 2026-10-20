from __future__ import annotations

import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import torch


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Mapping, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(data), indent=2, ensure_ascii=False), encoding="utf-8")


def append_csv(row: Mapping[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def bytes_to_mib(value: int | float) -> float:
    return float(value) / (1024.0**2)


def state_num_bytes(state: Mapping[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in state.values())


def tensor_dict_l2_norm(tensors: Mapping[str, torch.Tensor]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for tensor in tensors.values():
        x = tensor.detach().double().cpu()
        total += torch.sum(x * x)
    return float(torch.sqrt(total).item())


def relative_tensor_dict_error(
    lhs: Mapping[str, torch.Tensor],
    rhs: Mapping[str, torch.Tensor],
    eps: float = 1e-12,
) -> float:
    diff_sq = 0.0
    ref_sq = 0.0
    for name in lhs:
        a = lhs[name].detach().double().cpu()
        b = rhs[name].detach().double().cpu()
        diff_sq += float(torch.sum((a - b) ** 2).item())
        ref_sq += float(torch.sum(b**2).item())
    return math.sqrt(diff_sq) / max(math.sqrt(ref_sq), eps)


def clone_tensor_dict(tensors: Mapping[str, torch.Tensor], device: torch.device | str | None = None) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        copied = tensor.detach().clone()
        if device is not None:
            copied = copied.to(device)
        result[name] = copied
    return result


def flatten_named_tensors(tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if not tensors:
        return torch.empty(0)
    return torch.cat([tensor.detach().reshape(-1).cpu() for tensor in tensors.values()])
