from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import get_dataset_spec
from src.models import build_model
from flocora_baseline.flocora import LoRAConv2d, count_flocora_parameters, inject_flocora


def main() -> None:
    torch.manual_seed(42)
    spec = get_dataset_spec("cifar10")
    dense = build_model("resnet18", 10, "group", spec.image_size).eval()
    raw = sum(p.numel() for p in dense.parameters())
    reference = copy.deepcopy(dense).eval()

    inject_flocora(dense, rank=93, alpha=16 * 93)
    stats = count_flocora_parameters(dense, raw)

    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        y_ref = reference(x)
        y_lora = dense(x)
    initial_error = float((y_ref - y_lora).abs().max().item())
    if initial_error != 0.0:
        raise AssertionError(f"Initial adapter must be exactly zero; max error={initial_error}")

    for module in dense.modules():
        if isinstance(module, LoRAConv2d) and module.base.weight.requires_grad:
            raise AssertionError("Frozen base convolution unexpectedly requires gradients.")

    trainable = [p for p in dense.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=0.01, momentum=0.9)
    dense.train()
    labels = torch.tensor([0, 1, 2, 3])
    before = {
        name: p.detach().clone()
        for name, p in dense.named_parameters()
        if ".base.weight" in name
    }
    optimizer.zero_grad(set_to_none=True)
    loss = nn.CrossEntropyLoss()(dense(x), labels)
    loss.backward()
    optimizer.step()
    after = dict(dense.named_parameters())
    frozen_error = max(
        float((value - after[name].detach()).abs().max().item())
        for name, value in before.items()
    )
    if frozen_error != 0.0:
        raise AssertionError(f"Frozen base convolution changed; max error={frozen_error}")

    print("[PASS] Initial FLoCoRA output exactly equals the dense initialized model.")
    print("[PASS] Base convolution weights remain frozen after one optimizer step.")
    print(
        "[PASS] Communication count | "
        f"trainable={stats.trainable_parameters:,} / dense={stats.raw_dense_parameters:,} "
        f"= {stats.trainable_ratio_vs_dense:.6f}x | adapted_convs={stats.conv_layers_adapted}"
    )


if __name__ == "__main__":
    main()
