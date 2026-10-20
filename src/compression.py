from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch

from .projection import ProjectionSettings, decode_tensor, encode_tensor


@dataclass
class EncodedUpdate:
    tensors: Dict[str, torch.Tensor]
    original_shapes: Dict[str, torch.Size]
    payload_bytes: int


def encode_update_dict(
    updates: Mapping[str, torch.Tensor],
    settings: ProjectionSettings,
    noise_std: float,
    client_id: int,
    noise_seed: int | None = None,
) -> EncodedUpdate:
    encoded: Dict[str, torch.Tensor] = {}
    shapes: Dict[str, torch.Size] = {}
    payload_bytes = 0

    for name, tensor in updates.items():
        payload = encode_tensor(
            tensor=tensor,
            tensor_name=name,
            settings=settings,
            noise_std=noise_std,
            client_id=client_id,
            noise_seed=noise_seed,
        )
        encoded[name] = payload
        shapes[name] = tensor.shape
        payload_bytes += payload.numel() * payload.element_size()

    return EncodedUpdate(
        tensors=encoded,
        original_shapes=shapes,
        payload_bytes=payload_bytes,
    )


def decode_update_dict(
    encoded: Mapping[str, torch.Tensor],
    original_shapes: Mapping[str, torch.Size],
    settings: ProjectionSettings,
) -> Dict[str, torch.Tensor]:
    decoded: Dict[str, torch.Tensor] = {}
    for name, payload in encoded.items():
        decoded[name] = decode_tensor(
            encoded=payload,
            original_shape=original_shapes[name],
            tensor_name=name,
            settings=settings,
        )
    return decoded
