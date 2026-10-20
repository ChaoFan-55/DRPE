from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterator, Tuple

import torch


def _stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)


def _next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError("value must be positive")
    return 1 if value == 1 else 1 << (value - 1).bit_length()


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def fwht(vector: torch.Tensor) -> torch.Tensor:
    """Normalized fast Walsh-Hadamard transform for a one-dimensional tensor."""
    if vector.ndim != 1:
        raise ValueError("fwht expects a one-dimensional tensor.")
    n = vector.numel()
    if not _is_power_of_two(n):
        raise ValueError("fwht input length must be a power of two.")

    output = vector.clone()
    h = 1
    while h < n:
        shaped = output.reshape(-1, h * 2)
        left = shaped[:, :h].clone()
        right = shaped[:, h : 2 * h].clone()
        shaped[:, :h] = left + right
        shaped[:, h : 2 * h] = left - right
        output = shaped.reshape(-1)
        h *= 2
    return output / math.sqrt(n)


@dataclass(frozen=True)
class ProjectionSettings:
    compression_ratio: float
    block_size: int
    base_seed: int
    round_index: int
    dynamic: bool
    reconstruction_mode: str = "projected"
    device: str = "cpu"

    @property
    def effective_round(self) -> int:
        return self.round_index if self.dynamic else 0

    def validate(self) -> None:
        if not (0 < self.compression_ratio <= 1):
            raise ValueError("compression_ratio must be in (0, 1].")
        if not _is_power_of_two(self.block_size):
            raise ValueError("block_size must be a power of two.")
        if self.reconstruction_mode not in {"projected", "unbiased"}:
            raise ValueError("reconstruction_mode must be 'projected' or 'unbiased'.")


@dataclass(frozen=True)
class BlockLayout:
    start: int
    valid_dim: int
    transform_dim: int
    projected_dim: int
    block_index: int


def iter_block_layouts(numel: int, block_size: int, compression_ratio: float) -> Iterator[BlockLayout]:
    if numel < 1:
        return
    block_index = 0
    for start in range(0, numel, block_size):
        valid_dim = min(block_size, numel - start)
        if compression_ratio >= 1.0:
            transform_dim = valid_dim
            projected_dim = valid_dim
        else:
            transform_dim = _next_power_of_two(valid_dim)
            projected_dim = max(1, int(round(valid_dim * compression_ratio)))
            projected_dim = min(projected_dim, transform_dim)
        yield BlockLayout(
            start=start,
            valid_dim=valid_dim,
            transform_dim=transform_dim,
            projected_dim=projected_dim,
            block_index=block_index,
        )
        block_index += 1


def encoded_numel(numel: int, block_size: int, compression_ratio: float) -> int:
    return sum(layout.projected_dim for layout in iter_block_layouts(numel, block_size, compression_ratio))


def _make_signs_and_indices(
    settings: ProjectionSettings,
    tensor_name: str,
    layout: BlockLayout,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(settings.device)
    seed = _stable_seed(
        settings.base_seed,
        settings.effective_round,
        tensor_name,
        layout.block_index,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    signs = torch.randint(
        low=0,
        high=2,
        size=(layout.transform_dim,),
        generator=generator,
        device=device,
        dtype=torch.int64,
    ).to(torch.float32)
    signs.mul_(2.0).sub_(1.0)

    indices = torch.randperm(
        layout.transform_dim,
        generator=generator,
        device=device,
    )[: layout.projected_dim]
    return signs, indices


def _noise_for_block(
    shape: torch.Size,
    noise_std: float,
    settings: ProjectionSettings,
    tensor_name: str,
    block_index: int,
    client_id: int,
    noise_seed: int | None = None,
) -> torch.Tensor:
    device = torch.device(settings.device)
    if noise_std <= 0:
        return torch.zeros(shape, device=device, dtype=torch.float32)
    seed = _stable_seed(
        "noise",
        settings.base_seed if noise_seed is None else noise_seed,
        settings.effective_round,
        tensor_name,
        block_index,
        client_id,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, device=device, dtype=torch.float32) * noise_std


def encode_tensor(
    tensor: torch.Tensor,
    tensor_name: str,
    settings: ProjectionSettings,
    noise_std: float = 0.0,
    client_id: int = 0,
    noise_seed: int | None = None,
) -> torch.Tensor:
    """
    Encode a tensor with an implicit signed subsampled Hadamard projection.

    The projection matrix is P = S H D, whose rows are orthonormal. No dense
    matrix is materialized. At ratio=1 and noise_std=0, encoding is the identity.
    """
    settings.validate()
    device = torch.device(settings.device)
    flat = tensor.detach().to(device=device, dtype=torch.float32).reshape(-1)
    encoded_blocks: list[torch.Tensor] = []

    for layout in iter_block_layouts(flat.numel(), settings.block_size, settings.compression_ratio):
        block = flat[layout.start : layout.start + layout.valid_dim]

        if settings.compression_ratio >= 1.0:
            projected = block.clone()
        else:
            padded = torch.zeros(layout.transform_dim, device=device, dtype=torch.float32)
            padded[: layout.valid_dim] = block
            signs, indices = _make_signs_and_indices(settings, tensor_name, layout)
            transformed = fwht(padded * signs)
            projected = transformed[indices]

        projected = projected + _noise_for_block(
            projected.shape,
            noise_std=noise_std,
            settings=settings,
            tensor_name=tensor_name,
            block_index=layout.block_index,
            client_id=client_id,
            noise_seed=noise_seed,
        )
        encoded_blocks.append(projected)

    if not encoded_blocks:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(encoded_blocks).cpu()


def decode_tensor(
    encoded: torch.Tensor,
    original_shape: torch.Size | tuple[int, ...],
    tensor_name: str,
    settings: ProjectionSettings,
) -> torch.Tensor:
    """
    Decode with P^T. In projected mode the result is P^T P x. In unbiased mode
    each block is additionally multiplied by transform_dim/projected_dim so that
    the expected reconstruction over random subspaces equals the original block.
    """
    settings.validate()
    device = torch.device(settings.device)
    total_numel = math.prod(original_shape)
    encoded = encoded.detach().to(device=device, dtype=torch.float32).reshape(-1)
    restored_blocks: list[torch.Tensor] = []
    pointer = 0

    for layout in iter_block_layouts(total_numel, settings.block_size, settings.compression_ratio):
        block_encoded = encoded[pointer : pointer + layout.projected_dim]
        if block_encoded.numel() != layout.projected_dim:
            raise ValueError("Encoded payload length is inconsistent with the projection settings.")
        pointer += layout.projected_dim

        if settings.compression_ratio >= 1.0:
            restored = block_encoded.clone()
        else:
            signs, indices = _make_signs_and_indices(settings, tensor_name, layout)
            sparse = torch.zeros(layout.transform_dim, device=device, dtype=torch.float32)
            sparse[indices] = block_encoded
            restored_full = fwht(sparse) * signs
            if settings.reconstruction_mode == "unbiased":
                restored_full = restored_full * (layout.transform_dim / layout.projected_dim)
            restored = restored_full[: layout.valid_dim]
        restored_blocks.append(restored)

    if pointer != encoded.numel():
        raise ValueError("Encoded payload contains unexpected trailing values.")

    return torch.cat(restored_blocks).reshape(original_shape).cpu()
