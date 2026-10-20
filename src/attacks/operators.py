from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn

from ..projection import (
    BlockLayout,
    ProjectionSettings,
    _make_signs_and_indices,
    iter_block_layouts,
)


def differentiable_fwht(vector: torch.Tensor) -> torch.Tensor:
    """Out-of-place normalized FWHT for a one-dimensional tensor."""
    if vector.ndim != 1:
        raise ValueError("differentiable_fwht expects a one-dimensional tensor.")
    return differentiable_fwht_batch(vector.unsqueeze(0)).squeeze(0)


def differentiable_fwht_batch(matrix: torch.Tensor) -> torch.Tensor:
    """Normalized FWHT over the final dimension, preserving higher-order autograd.

    ``matrix`` may have any number of leading dimensions. Processing many DRPE
    blocks together avoids the original Python loop over every 512-value block.
    """
    if matrix.ndim < 1:
        raise ValueError("differentiable_fwht_batch expects at least one dimension.")
    n = matrix.shape[-1]
    if n < 1 or n & (n - 1):
        raise ValueError("FWHT input length must be a power of two.")

    output = matrix
    h = 1
    while h < n:
        prefix = output.shape[:-1]
        shaped = output.reshape(*prefix, -1, 2, h)
        left = shaped[..., 0, :]
        right = shaped[..., 1, :]
        output = torch.cat((left + right, left - right), dim=-1).reshape(*prefix, n)
        h *= 2
    return output / math.sqrt(n)


@lru_cache(maxsize=None)
def _cached_cpu_metadata(
    compression_ratio: float,
    block_size: int,
    base_seed: int,
    round_index: int,
    dynamic: bool,
    reconstruction_mode: str,
    projection_device: str,
    tensor_name: str,
    start: int,
    valid_dim: int,
    transform_dim: int,
    projected_dim: int,
    block_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate deterministic metadata once on the configured projection device.

    Training commonly uses ``projection_device='cpu'``. Keeping that device in
    the cache key ensures the attack reproduces the exact same projection stream.
    Returned values are moved to CPU before being cached, then transferred to the
    attack device in one batch rather than once per block and iteration.
    """
    settings = ProjectionSettings(
        compression_ratio=compression_ratio,
        block_size=block_size,
        base_seed=base_seed,
        round_index=round_index,
        dynamic=dynamic,
        reconstruction_mode=reconstruction_mode,
        device=projection_device,
    )
    layout = BlockLayout(
        start=start,
        valid_dim=valid_dim,
        transform_dim=transform_dim,
        projected_dim=projected_dim,
        block_index=block_index,
    )
    signs, indices = _make_signs_and_indices(settings, tensor_name, layout)
    return signs.detach().cpu(), indices.detach().cpu()


# Device copies are reused across inversion iterations. The cache is intentionally
# process-local: projection metadata is deterministic and small relative to the
# model gradients, while avoiding thousands of repeated CPU->GPU transfers.
_DEVICE_METADATA_CACHE: dict[tuple[object, ...], Tuple[torch.Tensor, torch.Tensor]] = {}
_BATCH_METADATA_CACHE: dict[tuple[object, ...], Tuple[torch.Tensor, torch.Tensor]] = {}


def clear_attack_projection_cache() -> None:
    """Release cached CPU/GPU projection metadata."""
    _DEVICE_METADATA_CACHE.clear()
    _BATCH_METADATA_CACHE.clear()
    _cached_cpu_metadata.cache_clear()


def _metadata_cpu(
    settings: ProjectionSettings,
    tensor_name: str,
    layout: BlockLayout,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _cached_cpu_metadata(
        settings.compression_ratio,
        settings.block_size,
        settings.base_seed,
        settings.round_index,
        settings.dynamic,
        settings.reconstruction_mode,
        settings.device,
        tensor_name,
        layout.start,
        layout.valid_dim,
        layout.transform_dim,
        layout.projected_dim,
        layout.block_index,
    )


def _metadata_on_tensor_device(
    settings: ProjectionSettings,
    tensor_name: str,
    layout: BlockLayout,
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (
        settings.compression_ratio,
        settings.block_size,
        settings.base_seed,
        settings.round_index,
        settings.dynamic,
        settings.reconstruction_mode,
        settings.device,
        tensor_name,
        layout.block_index,
        layout.valid_dim,
        layout.transform_dim,
        layout.projected_dim,
        str(tensor.device),
        tensor.dtype,
    )
    cached = _DEVICE_METADATA_CACHE.get(key)
    if cached is None:
        signs, indices = _metadata_cpu(settings, tensor_name, layout)
        cached = (
            signs.to(device=tensor.device, dtype=tensor.dtype),
            indices.to(device=tensor.device, dtype=torch.long),
        )
        _DEVICE_METADATA_CACHE[key] = cached
    return cached


def _full_block_metadata(
    settings: ProjectionSettings,
    tensor_name: str,
    full_block_count: int,
    projected_dim: int,
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (
        settings.compression_ratio,
        settings.block_size,
        settings.base_seed,
        settings.round_index,
        settings.dynamic,
        settings.reconstruction_mode,
        settings.device,
        tensor_name,
        full_block_count,
        projected_dim,
        str(tensor.device),
        tensor.dtype,
    )
    cached = _BATCH_METADATA_CACHE.get(key)
    if cached is not None:
        return cached

    sign_rows: list[torch.Tensor] = []
    index_rows: list[torch.Tensor] = []
    for block_index in range(full_block_count):
        layout = BlockLayout(
            start=block_index * settings.block_size,
            valid_dim=settings.block_size,
            transform_dim=settings.block_size,
            projected_dim=projected_dim,
            block_index=block_index,
        )
        signs, indices = _metadata_cpu(settings, tensor_name, layout)
        sign_rows.append(signs)
        index_rows.append(indices)

    cached = (
        torch.stack(sign_rows, dim=0).to(device=tensor.device, dtype=tensor.dtype),
        torch.stack(index_rows, dim=0).to(device=tensor.device, dtype=torch.long),
    )
    _BATCH_METADATA_CACHE[key] = cached
    return cached


def _projected_dim_for_full_block(settings: ProjectionSettings) -> int:
    layout = next(
        iter_block_layouts(
            settings.block_size,
            settings.block_size,
            settings.compression_ratio,
        )
    )
    return layout.projected_dim


def differentiable_encode_tensor(
    tensor: torch.Tensor,
    tensor_name: str,
    settings: ProjectionSettings,
    batch_blocks: int = 256,
) -> torch.Tensor:
    """Differentiable DRPE encoding with batched full-block processing."""
    settings.validate()
    if batch_blocks < 1:
        raise ValueError("batch_blocks must be positive.")

    flat = tensor.reshape(-1)
    if flat.numel() == 0:
        return torch.empty(0, device=tensor.device, dtype=tensor.dtype)
    if settings.compression_ratio >= 1.0:
        return flat

    block_size = settings.block_size
    full_block_count = flat.numel() // block_size
    tail_dim = flat.numel() % block_size
    encoded_parts: list[torch.Tensor] = []

    if full_block_count:
        projected_dim = _projected_dim_for_full_block(settings)
        signs, indices = _full_block_metadata(
            settings=settings,
            tensor_name=tensor_name,
            full_block_count=full_block_count,
            projected_dim=projected_dim,
            tensor=tensor,
        )
        full_blocks = flat[: full_block_count * block_size].reshape(full_block_count, block_size)
        for start in range(0, full_block_count, batch_blocks):
            end = min(start + batch_blocks, full_block_count)
            transformed = differentiable_fwht_batch(full_blocks[start:end] * signs[start:end])
            projected = torch.gather(transformed, dim=1, index=indices[start:end])
            encoded_parts.append(projected.reshape(-1))

    if tail_dim:
        layout = list(
            iter_block_layouts(flat.numel(), settings.block_size, settings.compression_ratio)
        )[-1]
        block = flat[layout.start : layout.start + layout.valid_dim]
        if layout.transform_dim == layout.valid_dim:
            padded = block
        else:
            padding = torch.zeros(
                layout.transform_dim - layout.valid_dim,
                dtype=block.dtype,
                device=block.device,
            )
            padded = torch.cat((block, padding), dim=0)
        signs, indices = _metadata_on_tensor_device(settings, tensor_name, layout, tensor)
        transformed = differentiable_fwht_batch((padded * signs).unsqueeze(0)).squeeze(0)
        encoded_parts.append(transformed.index_select(0, indices))

    return torch.cat(encoded_parts, dim=0)


def differentiable_decode_tensor(
    encoded: torch.Tensor,
    original_shape: torch.Size | tuple[int, ...],
    tensor_name: str,
    settings: ProjectionSettings,
    batch_blocks: int = 256,
) -> torch.Tensor:
    """Differentiable P^T decoding with batched full-block processing."""
    settings.validate()
    if batch_blocks < 1:
        raise ValueError("batch_blocks must be positive.")

    total_numel = math.prod(original_shape)
    encoded = encoded.reshape(-1)
    if total_numel == 0:
        return torch.empty(original_shape, device=encoded.device, dtype=encoded.dtype)
    if settings.compression_ratio >= 1.0:
        if encoded.numel() != total_numel:
            raise ValueError("Encoded payload length does not match identity projection.")
        return encoded.reshape(original_shape)

    block_size = settings.block_size
    full_block_count = total_numel // block_size
    tail_dim = total_numel % block_size
    restored_parts: list[torch.Tensor] = []
    pointer = 0

    if full_block_count:
        projected_dim = _projected_dim_for_full_block(settings)
        encoded_count = full_block_count * projected_dim
        full_encoded = encoded[pointer : pointer + encoded_count]
        if full_encoded.numel() != encoded_count:
            raise ValueError("Encoded payload length does not match full DRPE blocks.")
        pointer += encoded_count
        full_encoded = full_encoded.reshape(full_block_count, projected_dim)
        signs, indices = _full_block_metadata(
            settings=settings,
            tensor_name=tensor_name,
            full_block_count=full_block_count,
            projected_dim=projected_dim,
            tensor=encoded,
        )
        for start in range(0, full_block_count, batch_blocks):
            end = min(start + batch_blocks, full_block_count)
            sparse = torch.zeros(
                (end - start, block_size),
                dtype=encoded.dtype,
                device=encoded.device,
            )
            sparse = sparse.scatter(1, indices[start:end], full_encoded[start:end])
            restored = differentiable_fwht_batch(sparse) * signs[start:end]
            if settings.reconstruction_mode == "unbiased":
                restored = restored * (block_size / projected_dim)
            restored_parts.append(restored.reshape(-1))

    if tail_dim:
        layout = list(
            iter_block_layouts(total_numel, settings.block_size, settings.compression_ratio)
        )[-1]
        block_encoded = encoded[pointer : pointer + layout.projected_dim]
        if block_encoded.numel() != layout.projected_dim:
            raise ValueError("Encoded payload length does not match tail DRPE block.")
        pointer += layout.projected_dim
        signs, indices = _metadata_on_tensor_device(settings, tensor_name, layout, encoded)
        sparse = torch.zeros(layout.transform_dim, dtype=encoded.dtype, device=encoded.device)
        sparse = sparse.scatter(0, indices, block_encoded)
        restored_full = differentiable_fwht_batch(sparse.unsqueeze(0)).squeeze(0) * signs
        if settings.reconstruction_mode == "unbiased":
            restored_full = restored_full * (layout.transform_dim / layout.projected_dim)
        restored_parts.append(restored_full[: layout.valid_dim])

    if pointer != encoded.numel():
        raise ValueError("Encoded payload contains trailing values.")
    return torch.cat(restored_parts, dim=0).reshape(original_shape)


def differentiable_encode_dict(
    tensors: Mapping[str, torch.Tensor],
    settings: ProjectionSettings,
    batch_blocks: int = 256,
) -> Dict[str, torch.Tensor]:
    return {
        name: differentiable_encode_tensor(tensor, name, settings, batch_blocks=batch_blocks)
        for name, tensor in tensors.items()
    }


def differentiable_decode_dict(
    tensors: Mapping[str, torch.Tensor],
    original_shapes: Mapping[str, torch.Size | tuple[int, ...]],
    settings: ProjectionSettings,
    batch_blocks: int = 256,
) -> Dict[str, torch.Tensor]:
    return {
        name: differentiable_decode_tensor(
            tensor,
            original_shapes[name],
            name,
            settings,
            batch_blocks=batch_blocks,
        )
        for name, tensor in tensors.items()
    }


def select_named_parameters(
    model: nn.Module,
    scope: str,
) -> Sequence[tuple[str, nn.Parameter]]:
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if scope == "all":
        selected = named
    elif scope == "head_and_last_block":
        selected = [item for item in named if item[0].startswith("layer4.") or item[0].startswith("head.")]
    elif scope == "last_layer":
        selected = [item for item in named if item[0].startswith("head.")]
    else:
        raise ValueError(f"Unsupported parameter scope: {scope}")
    if not selected:
        raise ValueError(f"No parameters selected for scope={scope!r}.")
    return selected
