from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass(frozen=True)
class DenseRPSettings:
    compression_ratio: float
    block_size: int
    base_seed: int
    round_index: int
    dynamic: bool = True
    reconstruction_mode: str = "projected"
    device: str = "cuda"
    batch_blocks: int = 4096

    @property
    def effective_round(self) -> int:
        return self.round_index if self.dynamic else 0

    def validate(self) -> None:
        if not (0.0 < self.compression_ratio <= 1.0):
            raise ValueError("compression_ratio must be in (0, 1].")
        if self.block_size < 1:
            raise ValueError("block_size must be positive.")
        if self.reconstruction_mode not in {"projected", "unbiased"}:
            raise ValueError("reconstruction_mode must be projected/unbiased.")
        if self.batch_blocks < 1:
            raise ValueError("batch_blocks must be positive.")


@dataclass(frozen=True)
class BlockLayout:
    start: int
    valid_dim: int
    projected_dim: int
    block_index: int


_MATRIX_CACHE: dict[tuple, torch.Tensor] = {}
_SIGN_CACHE: dict[tuple, torch.Tensor] = {}


def clear_dense_rp_cache() -> None:
    """Clear per-round projection/sign caches."""
    _MATRIX_CACHE.clear()
    _SIGN_CACHE.clear()


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**63 - 1)


def resolve_projection_device(text: str) -> torch.device:
    text = text.lower()
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Dense RP requested CUDA projection but CUDA is unavailable.")
    return torch.device(text)


def projected_dim(valid_dim: int, compression_ratio: float) -> int:
    if compression_ratio >= 1.0:
        return valid_dim
    return min(valid_dim, max(1, int(round(valid_dim * compression_ratio))))


def iter_block_layouts(
    numel: int,
    block_size: int,
    compression_ratio: float,
) -> Iterator[BlockLayout]:
    if numel < 1:
        return
    block_index = 0
    for start in range(0, numel, block_size):
        valid_dim = min(block_size, numel - start)
        yield BlockLayout(
            start=start,
            valid_dim=valid_dim,
            projected_dim=projected_dim(valid_dim, compression_ratio),
            block_index=block_index,
        )
        block_index += 1


def encoded_numel(numel: int, block_size: int, compression_ratio: float) -> int:
    return sum(
        layout.projected_dim
        for layout in iter_block_layouts(numel, block_size, compression_ratio)
    )


def _orthonormal_dense_matrix(
    valid_dim: int,
    projected_dim_value: int,
    settings: DenseRPSettings,
) -> torch.Tensor:
    """
    Return P in R^{m x n} with orthonormal rows.

    Construction:
        A ~ N(0, I), A in R^{n x m}
        Q, _ = qr(A)
        P = Q^T

    The matrix is generated on CPU for deterministic construction and cached,
    then moved to the requested projection device.
    """
    device = resolve_projection_device(settings.device)
    key = (
        settings.base_seed,
        settings.effective_round,
        valid_dim,
        projected_dim_value,
        str(device),
    )
    cached = _MATRIX_CACHE.get(key)
    if cached is not None:
        return cached

    if projected_dim_value == valid_dim:
        matrix_cpu = torch.eye(valid_dim, dtype=torch.float32)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            stable_seed(
                "dense_rp_matrix",
                settings.base_seed,
                settings.effective_round,
                valid_dim,
                projected_dim_value,
            )
        )
        gaussian = torch.randn(
            (valid_dim, projected_dim_value),
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        # reduced QR -> Q has orthonormal columns; Q^T has orthonormal rows.
        q, _ = torch.linalg.qr(gaussian, mode="reduced")
        matrix_cpu = q.transpose(0, 1).contiguous()

    matrix = matrix_cpu.to(device=device, dtype=torch.float32)
    _MATRIX_CACHE[key] = matrix
    return matrix


def _tensor_signs(
    tensor_name: str,
    valid_dim: int,
    settings: DenseRPSettings,
) -> torch.Tensor:
    """
    Tensor-specific sign diagonal D. The dense projection actually used is P D.
    This keeps rows orthonormal while avoiding exactly the same subspace for
    every parameter tensor.
    """
    device = resolve_projection_device(settings.device)
    key = (
        settings.base_seed,
        settings.effective_round,
        tensor_name,
        valid_dim,
        str(device),
    )
    cached = _SIGN_CACHE.get(key)
    if cached is not None:
        return cached

    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        stable_seed(
            "dense_rp_signs",
            settings.base_seed,
            settings.effective_round,
            tensor_name,
            valid_dim,
        )
    )
    signs_cpu = torch.randint(
        low=0,
        high=2,
        size=(valid_dim,),
        generator=generator,
        dtype=torch.int64,
    ).float()
    signs_cpu = signs_cpu.mul_(2.0).sub_(1.0)
    signs = signs_cpu.to(device=device, dtype=torch.float32)
    _SIGN_CACHE[key] = signs
    return signs


def _encode_same_dim_blocks(
    blocks_cpu: torch.Tensor,
    tensor_name: str,
    valid_dim: int,
    projected_dim_value: int,
    settings: DenseRPSettings,
) -> torch.Tensor:
    device = resolve_projection_device(settings.device)
    p = _orthonormal_dense_matrix(valid_dim, projected_dim_value, settings)
    signs = _tensor_signs(tensor_name, valid_dim, settings)

    outputs: list[torch.Tensor] = []
    for start in range(0, blocks_cpu.shape[0], settings.batch_blocks):
        batch = blocks_cpu[start : start + settings.batch_blocks].to(
            device=device,
            dtype=torch.float32,
        )
        projected = (batch * signs) @ p.transpose(0, 1)
        outputs.append(projected.detach().cpu())
        del batch, projected

    if not outputs:
        return torch.empty((0, projected_dim_value), dtype=torch.float32)
    return torch.cat(outputs, dim=0)


def encode_tensor(
    tensor: torch.Tensor,
    tensor_name: str,
    settings: DenseRPSettings,
) -> torch.Tensor:
    """
    Block-wise dense orthonormal random projection.

    For ratio < 1:
        y = P_t D_{t,name} x

    where P_t is dense and row-orthonormal. No clipping, no noise and no
    privacy mechanism are applied.
    """
    settings.validate()
    flat = tensor.detach().float().cpu().reshape(-1)

    if flat.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    if settings.compression_ratio >= 1.0:
        return flat.clone()

    block_size = settings.block_size
    num_full = flat.numel() // block_size
    remainder = flat.numel() % block_size

    pieces: list[torch.Tensor] = []

    if num_full:
        full = flat[: num_full * block_size].reshape(num_full, block_size)
        m = projected_dim(block_size, settings.compression_ratio)
        encoded_full = _encode_same_dim_blocks(
            full,
            tensor_name=tensor_name,
            valid_dim=block_size,
            projected_dim_value=m,
            settings=settings,
        )
        pieces.append(encoded_full.reshape(-1))

    if remainder:
        tail = flat[num_full * block_size :].reshape(1, remainder)
        m_tail = projected_dim(remainder, settings.compression_ratio)
        encoded_tail = _encode_same_dim_blocks(
            tail,
            tensor_name=tensor_name,
            valid_dim=remainder,
            projected_dim_value=m_tail,
            settings=settings,
        )
        pieces.append(encoded_tail.reshape(-1))

    return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.float32)


def _decode_same_dim_blocks(
    encoded_cpu: torch.Tensor,
    tensor_name: str,
    valid_dim: int,
    projected_dim_value: int,
    settings: DenseRPSettings,
) -> torch.Tensor:
    device = resolve_projection_device(settings.device)
    p = _orthonormal_dense_matrix(valid_dim, projected_dim_value, settings)
    signs = _tensor_signs(tensor_name, valid_dim, settings)

    outputs: list[torch.Tensor] = []
    for start in range(0, encoded_cpu.shape[0], settings.batch_blocks):
        batch = encoded_cpu[start : start + settings.batch_blocks].to(
            device=device,
            dtype=torch.float32,
        )
        restored = (batch @ p) * signs

        if settings.reconstruction_mode == "unbiased":
            restored = restored * (valid_dim / projected_dim_value)

        outputs.append(restored.detach().cpu())
        del batch, restored

    if not outputs:
        return torch.empty((0, valid_dim), dtype=torch.float32)
    return torch.cat(outputs, dim=0)


def decode_tensor(
    encoded: torch.Tensor,
    original_shape: torch.Size | tuple[int, ...],
    tensor_name: str,
    settings: DenseRPSettings,
) -> torch.Tensor:
    """
    Decode with (P D)^T.

    projected mode:
        x_hat = D P^T P D x

    unbiased mode:
        multiply each block by n/m after decoding.
    """
    settings.validate()
    total_numel = math.prod(original_shape)
    encoded = encoded.detach().float().cpu().reshape(-1)

    if total_numel == 0:
        return torch.empty(original_shape, dtype=torch.float32)
    if settings.compression_ratio >= 1.0:
        if encoded.numel() != total_numel:
            raise ValueError("ratio=1 payload length mismatch.")
        return encoded.reshape(original_shape).clone()

    block_size = settings.block_size
    num_full = total_numel // block_size
    remainder = total_numel % block_size
    pointer = 0
    pieces: list[torch.Tensor] = []

    if num_full:
        m = projected_dim(block_size, settings.compression_ratio)
        count = num_full * m
        full_encoded = encoded[pointer : pointer + count].reshape(num_full, m)
        pointer += count
        decoded_full = _decode_same_dim_blocks(
            full_encoded,
            tensor_name=tensor_name,
            valid_dim=block_size,
            projected_dim_value=m,
            settings=settings,
        )
        pieces.append(decoded_full.reshape(-1))

    if remainder:
        m_tail = projected_dim(remainder, settings.compression_ratio)
        tail_encoded = encoded[pointer : pointer + m_tail].reshape(1, m_tail)
        pointer += m_tail
        decoded_tail = _decode_same_dim_blocks(
            tail_encoded,
            tensor_name=tensor_name,
            valid_dim=remainder,
            projected_dim_value=m_tail,
            settings=settings,
        )
        pieces.append(decoded_tail.reshape(-1))

    if pointer != encoded.numel():
        raise ValueError(
            f"Encoded payload length mismatch for {tensor_name}: "
            f"used {pointer}, got {encoded.numel()}."
        )

    return torch.cat(pieces).reshape(original_shape)
def differentiable_encode_tensor(
    tensor: torch.Tensor,
    tensor_name: str,
    settings: DenseRPSettings,
) -> torch.Tensor:
    """
    Differentiable version of Dense-RP encoding for white-box inversion attacks.
    It uses exactly the same P and tensor-specific D as encode_tensor, but does
    not detach the candidate tensor.
    """
    settings.validate()
    flat = tensor.reshape(-1)

    if flat.numel() == 0:
        return flat
    if settings.compression_ratio >= 1.0:
        return flat

    device = flat.device
    projection_device = resolve_projection_device(settings.device)

    # Normalize CUDA device notation:
    # torch.device("cuda") and torch.device("cuda:0") refer to the same
    # default GPU on a single-GPU system.
    if projection_device.type == "cuda" and projection_device.index is None:
        projection_device = torch.device(
            "cuda",
            torch.cuda.current_device(),
        )

    if device.type == "cuda" and device.index is None:
        device = torch.device(
            "cuda",
            torch.cuda.current_device(),
        )

    if device != projection_device:
        raise ValueError(
            f"Differentiable Dense-RP expects candidate tensor on {projection_device}, "
            f"got {device}."
        )

    block_size = settings.block_size
    num_full = flat.numel() // block_size
    remainder = flat.numel() % block_size
    pieces: list[torch.Tensor] = []

    if num_full:
        full = flat[: num_full * block_size].reshape(num_full, block_size)
        m = projected_dim(block_size, settings.compression_ratio)
        p = _orthonormal_dense_matrix(block_size, m, settings)
        signs = _tensor_signs(tensor_name, block_size, settings)
        outputs: list[torch.Tensor] = []
        for start in range(0, num_full, settings.batch_blocks):
            batch = full[start : start + settings.batch_blocks]
            outputs.append((batch * signs) @ p.transpose(0, 1))
        pieces.append(torch.cat(outputs, dim=0).reshape(-1))

    if remainder:
        tail = flat[num_full * block_size :].reshape(1, remainder)
        m_tail = projected_dim(remainder, settings.compression_ratio)
        p_tail = _orthonormal_dense_matrix(remainder, m_tail, settings)
        signs_tail = _tensor_signs(tensor_name, remainder, settings)
        pieces.append(((tail * signs_tail) @ p_tail.transpose(0, 1)).reshape(-1))

    return torch.cat(pieces) if pieces else flat.new_empty(0)
