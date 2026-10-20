from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PowerSGDSettings:
    rank: int
    base_seed: int
    round_index: int
    client_id: int
    power_iterations: int = 1
    device: str = "cuda"
    fallback_dense_if_not_smaller: bool = True

    def validate(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be positive.")
        if self.power_iterations < 1:
            raise ValueError("power_iterations must be at least 1.")


_Q0_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}


def clear_powersgd_cache() -> None:
    _Q0_CACHE.clear()


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**63 - 1)


def resolve_device(text: str) -> torch.device:
    text = text.lower()
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PowerSGD requested CUDA but CUDA is unavailable.")
    device = torch.device(text)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def matrix_shape(shape: torch.Size | tuple[int, ...]) -> tuple[int, int]:
    if len(shape) < 2:
        raise ValueError("PowerSGD matrix_shape requires ndim >= 2.")
    rows = int(shape[0])
    total = math.prod(shape)
    cols = total // rows
    return rows, cols


def effective_rank(
    shape: torch.Size | tuple[int, ...],
    configured_rank: int,
) -> int:
    rows, cols = matrix_shape(shape)
    return min(configured_rank, rows, cols)


def should_use_low_rank(
    shape: torch.Size | tuple[int, ...],
    rank: int,
    fallback_dense_if_not_smaller: bool = True,
) -> bool:
    if len(shape) < 2:
        return False
    rows, cols = matrix_shape(shape)
    r = min(rank, rows, cols)
    factor_count = r * (rows + cols)
    dense_count = rows * cols
    if fallback_dense_if_not_smaller and factor_count >= dense_count:
        return False
    return True


def payload_numel_for_shape(
    shape: torch.Size | tuple[int, ...],
    rank: int,
    fallback_dense_if_not_smaller: bool = True,
) -> int:
    total = math.prod(shape)
    if not should_use_low_rank(shape, rank, fallback_dense_if_not_smaller):
        return total
    rows, cols = matrix_shape(shape)
    r = min(rank, rows, cols)
    return r * (rows + cols)


def _orthogonalize(matrix: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q


def _initial_q(
    tensor_name: str,
    rows: int,
    cols: int,
    rank: int,
    settings: PowerSGDSettings,
    device: torch.device,
) -> torch.Tensor:
    """
    Deterministic random initialization Q0. It is not transmitted; both the
    compressor and the white-box attack reconstruct it from public settings.
    """
    key = (
        settings.base_seed,
        settings.round_index,
        settings.client_id,
        tensor_name,
        rows,
        cols,
        rank,
        str(device),
    )
    cached = _Q0_CACHE.get(key)
    if cached is not None:
        return cached

    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        stable_seed(
            "powersgd_q0",
            settings.base_seed,
            settings.round_index,
            settings.client_id,
            tensor_name,
            rows,
            cols,
            rank,
        )
    )
    q0_cpu = torch.randn((cols, rank), generator=generator, dtype=torch.float32)
    q0 = _orthogonalize(q0_cpu.to(device=device))
    _Q0_CACHE[key] = q0
    return q0


def _factorize_matrix(
    matrix: torch.Tensor,
    tensor_name: str,
    settings: PowerSGDSettings,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    One-step (or multi-step) power iteration.

    For the default power_iterations=1:
      Q0 <- random orthonormal
      P  <- orth(M Q0)
      Q  <- M^T P
      M_hat = P Q^T

    P and Q are the transmitted factors.
    """
    settings.validate()
    device = resolve_device(settings.device)
    matrix = matrix.to(device=device, dtype=torch.float32)

    rows, cols = matrix.shape
    rank = min(settings.rank, rows, cols)
    q = _initial_q(
        tensor_name=tensor_name,
        rows=rows,
        cols=cols,
        rank=rank,
        settings=settings,
        device=device,
    )

    p = None
    for iteration in range(settings.power_iterations):
        p = _orthogonalize(matrix @ q)
        q = matrix.transpose(0, 1) @ p
        if iteration < settings.power_iterations - 1:
            q = _orthogonalize(q)

    assert p is not None
    return p, q


def encode_tensor(
    tensor: torch.Tensor,
    tensor_name: str,
    settings: PowerSGDSettings,
) -> dict[str, Any]:
    """
    Compress one model-update tensor.

    1D tensors and matrices for which factorization would not save bytes are
    sent dense. Matrix-like tensors are reshaped as:
        [shape[0], numel / shape[0]]
    and compressed to P, Q.
    """
    settings.validate()
    shape = tensor.shape

    if not should_use_low_rank(
        shape,
        settings.rank,
        settings.fallback_dense_if_not_smaller,
    ):
        value = tensor.detach().float().cpu().clone()
        return {
            "kind": "dense",
            "shape": tuple(shape),
            "value": value,
        }

    rows, cols = matrix_shape(shape)
    matrix = tensor.detach().float().cpu().reshape(rows, cols)
    p, q = _factorize_matrix(matrix, tensor_name, settings)

    return {
        "kind": "lowrank",
        "shape": tuple(shape),
        "rows": rows,
        "cols": cols,
        "rank": int(p.shape[1]),
        "p": p.detach().cpu(),
        "q": q.detach().cpu(),
    }


def payload_numel(payload: dict[str, Any]) -> int:
    if payload["kind"] == "dense":
        return int(payload["value"].numel())
    return int(payload["p"].numel() + payload["q"].numel())


def payload_bytes(payload: dict[str, Any]) -> int:
    if payload["kind"] == "dense":
        value = payload["value"]
        return int(value.numel() * value.element_size())
    p = payload["p"]
    q = payload["q"]
    return int(p.numel() * p.element_size() + q.numel() * q.element_size())


def decode_tensor(payload: dict[str, Any]) -> torch.Tensor:
    if payload["kind"] == "dense":
        return payload["value"].detach().float().cpu().reshape(payload["shape"])

    p = payload["p"].detach().float().cpu()
    q = payload["q"].detach().float().cpu()
    restored = p @ q.transpose(0, 1)
    return restored.reshape(payload["shape"])


def target_payload_to_device(
    payload: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    if payload["kind"] == "dense":
        return {
            "kind": "dense",
            "shape": payload["shape"],
            "value": payload["value"].to(device=device, dtype=torch.float32),
        }
    return {
        "kind": "lowrank",
        "shape": payload["shape"],
        "rows": payload["rows"],
        "cols": payload["cols"],
        "rank": payload["rank"],
        "p": payload["p"].to(device=device, dtype=torch.float32),
        "q": payload["q"].to(device=device, dtype=torch.float32),
    }


def differentiable_encode_tensor(
    tensor: torch.Tensor,
    tensor_name: str,
    settings: PowerSGDSettings,
) -> dict[str, Any]:
    """
    Differentiable PowerSGD encoding for white-box inversion attacks.
    Uses the exact same deterministic Q0 and factorization rule as encode_tensor.
    """
    settings.validate()
    shape = tensor.shape

    if not should_use_low_rank(
        shape,
        settings.rank,
        settings.fallback_dense_if_not_smaller,
    ):
        return {
            "kind": "dense",
            "shape": tuple(shape),
            "value": tensor,
        }

    device = tensor.device
    projection_device = resolve_device(settings.device)
    if projection_device.type == "cuda" and device.type == "cuda":
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        projection_index = projection_device.index
        if projection_index is None:
            projection_index = torch.cuda.current_device()
        if device_index != projection_index:
            raise ValueError(
                f"Candidate tensor is on cuda:{device_index}, "
                f"PowerSGD is on cuda:{projection_index}."
            )
    elif device.type != projection_device.type:
        raise ValueError(
            f"Candidate tensor is on {device}, PowerSGD is on {projection_device}."
        )

    rows, cols = matrix_shape(shape)
    matrix = tensor.reshape(rows, cols)
    rank = min(settings.rank, rows, cols)
    q = _initial_q(
        tensor_name=tensor_name,
        rows=rows,
        cols=cols,
        rank=rank,
        settings=settings,
        device=matrix.device,
    )

    p = None
    for iteration in range(settings.power_iterations):
        p = _orthogonalize(matrix @ q)
        q = matrix.transpose(0, 1) @ p
        if iteration < settings.power_iterations - 1:
            q = _orthogonalize(q)

    assert p is not None
    return {
        "kind": "lowrank",
        "shape": tuple(shape),
        "rows": rows,
        "cols": cols,
        "rank": int(rank),
        "p": p,
        "q": q,
    }


def decoded_payload_on_device(
    payload: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    if payload["kind"] == "dense":
        return payload["value"].to(device=device, dtype=torch.float32).reshape(payload["shape"])
    p = payload["p"].to(device=device, dtype=torch.float32)
    q = payload["q"].to(device=device, dtype=torch.float32)
    return (p @ q.transpose(0, 1)).reshape(payload["shape"])
