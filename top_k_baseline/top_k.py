from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SparseTensor:
    """One sparse tensor transmitted as int32 indices + float32 values."""
    indices: torch.Tensor
    values: torch.Tensor
    shape: torch.Size
    numel: int

    @property
    def selected_count(self) -> int:
        return int(self.indices.numel())

    @property
    def payload_bytes(self) -> int:
        return int(
            self.indices.numel() * self.indices.element_size()
            + self.values.numel() * self.values.element_size()
        )


def selected_count(numel: int, keep_ratio: float) -> int:
    if numel < 1:
        raise ValueError("numel must be positive.")
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError("keep_ratio must be in (0, 1].")
    return min(numel, max(1, int(round(numel * keep_ratio))))


def encode_top_k_tensor(tensor: torch.Tensor, keep_ratio: float) -> SparseTensor:
    """
    Tensor-wise Top-k compressor.

    Keeps the k largest-magnitude entries of a tensor. The transmitted
    representation is:
        - values: float32
        - indices: int32

    No clipping, noise, rescaling, or error feedback is applied.
    """
    flat = tensor.detach().float().cpu().reshape(-1)
    n = int(flat.numel())
    k = selected_count(n, keep_ratio)

    if k == n:
        indices64 = torch.arange(n, dtype=torch.long)
    else:
        indices64 = torch.topk(
            flat.abs(),
            k=k,
            largest=True,
            sorted=False,
        ).indices

    values = flat.index_select(0, indices64).clone()
    indices = indices64.to(dtype=torch.int32)

    return SparseTensor(
        indices=indices,
        values=values,
        shape=tensor.shape,
        numel=n,
    )


def gather_on_observed_support(
    tensor: torch.Tensor,
    sparse_reference: SparseTensor,
) -> torch.Tensor:
    """
    Differentiably gather a dense candidate tensor at the indices actually
    observed in a Top-k upload. Used by the white-box inversion attack.
    """
    flat = tensor.reshape(-1)
    indices = sparse_reference.indices.to(device=flat.device, dtype=torch.long)
    return flat.index_select(0, indices)
