from __future__ import annotations

import torch

from src.compression import decode_update_dict, encode_update_dict
from src.projection import ProjectionSettings


def test_ratio_one_encoded_aggregation_matches_raw_fedavg() -> None:
    torch.manual_seed(3)
    update_a = {"w": torch.randn(7, 9), "b": torch.randn(7)}
    update_b = {"w": torch.randn(7, 9), "b": torch.randn(7)}
    weight_a, weight_b = 0.3, 0.7

    settings = ProjectionSettings(
        compression_ratio=1.0,
        block_size=512,
        base_seed=99,
        round_index=1,
        dynamic=True,
        reconstruction_mode="projected",
        device="cpu",
    )
    enc_a = encode_update_dict(update_a, settings, noise_std=0.0, client_id=0)
    enc_b = encode_update_dict(update_b, settings, noise_std=0.0, client_id=1)
    averaged_encoded = {
        name: weight_a * enc_a.tensors[name] + weight_b * enc_b.tensors[name]
        for name in enc_a.tensors
    }
    decoded = decode_update_dict(averaged_encoded, enc_a.original_shapes, settings)
    raw = {
        name: weight_a * update_a[name] + weight_b * update_b[name]
        for name in update_a
    }

    for name in raw:
        assert torch.allclose(decoded[name], raw[name], atol=1e-7, rtol=1e-7)
