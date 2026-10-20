from __future__ import annotations

import torch

from src.projection import ProjectionSettings, decode_tensor, encode_tensor


def test_ratio_one_is_identity() -> None:
    torch.manual_seed(7)
    tensor = torch.randn(13, 17)
    settings = ProjectionSettings(
        compression_ratio=1.0,
        block_size=512,
        base_seed=123,
        round_index=3,
        dynamic=True,
        reconstruction_mode="projected",
        device="cpu",
    )
    encoded = encode_tensor(tensor, "weight", settings, noise_std=0.0, client_id=0)
    decoded = decode_tensor(encoded, tensor.shape, "weight", settings)
    assert torch.allclose(decoded, tensor, atol=1e-7, rtol=1e-7)


def test_dynamic_round_changes_payload() -> None:
    tensor = torch.arange(1024, dtype=torch.float32)
    settings_1 = ProjectionSettings(0.2, 512, 123, 1, True, "projected", "cpu")
    settings_2 = ProjectionSettings(0.2, 512, 123, 2, True, "projected", "cpu")
    y1 = encode_tensor(tensor, "weight", settings_1)
    y2 = encode_tensor(tensor, "weight", settings_2)
    assert y1.shape == y2.shape
    assert not torch.allclose(y1, y2)


def test_static_round_reuses_payload() -> None:
    tensor = torch.arange(1024, dtype=torch.float32)
    settings_1 = ProjectionSettings(0.2, 512, 123, 1, False, "projected", "cpu")
    settings_2 = ProjectionSettings(0.2, 512, 123, 2, False, "projected", "cpu")
    y1 = encode_tensor(tensor, "weight", settings_1)
    y2 = encode_tensor(tensor, "weight", settings_2)
    assert torch.allclose(y1, y2)
