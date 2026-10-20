from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn

from src.client import ClientResult
from src.state import load_aggregated_buffers


# ============================================================
# Data structures
# ============================================================

@dataclass
class SparseTensor:
    indices: torch.Tensor
    values: torch.Tensor


@dataclass
class RandomKClientResult:
    client_id: int
    num_samples: int

    payload: Dict[str, SparseTensor]

    floating_buffers: Dict[str, torch.Tensor]
    integer_buffers: Dict[str, torch.Tensor]

    local_loss: float
    update_norm: float

    selected_numel: int
    original_numel: int

    payload_bytes: int
    buffer_bytes: int
    uplink_bytes: int


@dataclass
class RandomKAggregationResult:
    decoded_update_norm: float
    total_uplink_bytes: int
    mean_client_loss: float
    mean_update_norm: float
    selected_fraction: float


# ============================================================
# Stable seed
# ============================================================

def stable_seed(*parts: object) -> int:
    """
    Generate a reproducible seed independent of Python hash().
    """

    text = "|".join(str(part) for part in parts)

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    ) % (2**63 - 1)


# ============================================================
# Random-k sampling
# ============================================================

def sample_random_k_indices(
    numel: int,
    k: int,
    seed: int,
    sampling_device: torch.device,
) -> torch.Tensor:
    """
    Select exactly k coordinates without replacement.

    When CUDA is available, randperm is generated on GPU.
    Only the selected k indices are copied back to CPU.

    This avoids the large CPU-memory peak caused by generating
    the complete permutation on CPU.
    """

    if k >= numel:
        return torch.arange(
            numel,
            dtype=torch.int64,
            device="cpu",
        )

    if sampling_device.type == "cuda":

        generator = torch.Generator(
            device="cuda"
        )
        generator.manual_seed(seed)

        permutation = torch.randperm(
            numel,
            generator=generator,
            device=sampling_device,
            dtype=torch.int64,
        )

        selected = (
            permutation[:k]
            .cpu()
        )

        del permutation

        torch.cuda.empty_cache()

        return selected

    # CPU fallback
    generator = torch.Generator(
        device="cpu"
    )
    generator.manual_seed(seed)

    return torch.randperm(
        numel,
        generator=generator,
        dtype=torch.int64,
    )[:k]


# ============================================================
# Client-side Random-k compression
# ============================================================

def compress_client_result(
    result: ClientResult,
    ratio: float,
    base_seed: int,
    round_index: int,
    unbiased: bool,
    sampling_device: torch.device,
) -> RandomKClientResult:
    """
    Convert the dense FedAvg update returned by FederatedClient
    into a Random-k sparse payload.

    Standard unbiased Random-k:

        Q(x)_j = x_j / p, if j is selected
                 0,       otherwise

    where p = k / d.
    """

    sparse_payload: Dict[str, SparseTensor] = {}

    payload_bytes = 0
    selected_numel = 0
    original_numel = 0

    # Process tensors one at a time.
    # pop() allows the dense update tensor to be released immediately.
    parameter_names = list(
        result.payload.keys()
    )

    for name in parameter_names:

        tensor = result.payload.pop(name)

        flat = (
            tensor.detach()
            .reshape(-1)
            .float()
            .cpu()
        )

        n = int(flat.numel())

        k = max(
            1,
            int(round(n * ratio)),
        )

        k = min(k, n)

        seed = stable_seed(
            base_seed,
            round_index,
            result.client_id,
            name,
        )

        indices64 = sample_random_k_indices(
            numel=n,
            k=k,
            seed=seed,
            sampling_device=sampling_device,
        )

        values = (
            flat.index_select(
                0,
                indices64,
            )
            .clone()
            .float()
        )

        # Unbiased scaling
        if unbiased and k < n:
            values.mul_(
                n / k
            )

        # Communication uses int32 indices.
        indices32 = indices64.to(
            dtype=torch.int32
        )

        sparse_payload[name] = SparseTensor(
            indices=indices32,
            values=values,
        )

        # float32 value = 4 bytes
        # int32 index   = 4 bytes
        payload_bytes += (
            values.numel()
            * values.element_size()
        )

        payload_bytes += (
            indices32.numel()
            * indices32.element_size()
        )

        selected_numel += k
        original_numel += n

        del indices64
        del flat
        del tensor

    # Keep the same buffer communication accounting
    # as the original FL implementation.
    buffer_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in result.floating_buffers.values()
    )

    buffer_bytes += sum(
        tensor.numel() * tensor.element_size()
        for tensor in result.integer_buffers.values()
    )

    if sampling_device.type == "cuda":
        torch.cuda.empty_cache()

    return RandomKClientResult(
        client_id=result.client_id,
        num_samples=result.num_samples,

        payload=sparse_payload,

        floating_buffers=result.floating_buffers,
        integer_buffers=result.integer_buffers,

        local_loss=result.local_loss,
        update_norm=result.update_norm,

        selected_numel=selected_numel,
        original_numel=original_numel,

        payload_bytes=payload_bytes,
        buffer_bytes=buffer_bytes,

        uplink_bytes=(
            payload_bytes
            + buffer_bytes
        ),
    )


# ============================================================
# Server-side sparse aggregation
# ============================================================

@torch.no_grad()
def aggregate_random_k(
    model: nn.Module,
    results: List[RandomKClientResult],
    server_lr: float,
) -> RandomKAggregationResult:
    """
    Sample-size weighted Random-k aggregation.

    The global model is expected to remain on CPU.

    Sparse updates are accumulated one parameter tensor at a time,
    avoiding reconstruction of multiple complete dense client
    updates simultaneously.
    """

    if not results:
        raise ValueError(
            "Cannot aggregate empty client results."
        )

    total_samples = sum(
        result.num_samples
        for result in results
    )

    decoded_norm_sq = 0.0

    # ========================================================
    # Aggregate parameter updates
    # ========================================================

    for name, parameter in model.named_parameters():

        accumulator = torch.zeros(
            parameter.numel(),
            dtype=torch.float32,
            device="cpu",
        )

        for result in results:

            sparse = result.payload[name]

            indices = (
                sparse.indices
                .long()
            )

            weight = (
                result.num_samples
                / total_samples
            )

            accumulator.index_add_(
                0,
                indices,
                sparse.values,
                alpha=weight,
            )

            del indices

        # Only for logging.
        tensor_norm = float(
            torch.linalg.vector_norm(
                accumulator
            ).item()
        )

        decoded_norm_sq += (
            tensor_norm
            * tensor_norm
        )

        update = accumulator.reshape(
            parameter.shape
        )

        parameter.add_(
            update,
            alpha=server_lr,
        )

        del update
        del accumulator

    # ========================================================
    # Aggregate buffers
    # Same logic as original server
    # ========================================================

    floating_buffers: Dict[
        str,
        torch.Tensor,
    ] = {}

    integer_buffers: Dict[
        str,
        torch.Tensor,
    ] = {}

    if results[0].floating_buffers:

        for name in results[0].floating_buffers:

            reference = (
                results[0]
                .floating_buffers[name]
            )

            accumulator = torch.zeros_like(
                reference,
                dtype=torch.float32,
            )

            for result in results:

                weight = (
                    result.num_samples
                    / total_samples
                )

                accumulator.add_(
                    result
                    .floating_buffers[name]
                    .float(),
                    alpha=weight,
                )

            floating_buffers[name] = accumulator

    if results[0].integer_buffers:

        for name in results[0].integer_buffers:

            stacked = torch.stack([
                result.integer_buffers[name].long()
                for result in results
            ])

            integer_buffers[name] = (
                torch.max(
                    stacked,
                    dim=0,
                ).values
            )

    load_aggregated_buffers(
        model,
        floating_buffers,
        integer_buffers,
    )

    # ========================================================
    # Statistics
    # ========================================================

    total_selected = sum(
        result.selected_numel
        for result in results
    )

    total_original = sum(
        result.original_numel
        for result in results
    )

    return RandomKAggregationResult(
        decoded_update_norm=(
            decoded_norm_sq ** 0.5
        ),

        total_uplink_bytes=sum(
            result.uplink_bytes
            for result in results
        ),

        mean_client_loss=(
            sum(
                result.local_loss
                for result in results
            )
            / len(results)
        ),

        mean_update_norm=(
            sum(
                result.update_norm
                for result in results
            )
            / len(results)
        ),

        selected_fraction=(
            total_selected
            / max(total_original, 1)
        ),
    )