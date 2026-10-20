from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAConv2d(nn.Module):
    """
    FLoCoRA convolution adapter in the paper's notation.

    Frozen base convolution:
        W0 : [O, I, K, K]

    Trainable adapter:
        B (down) : [r, I, K, K]
        A (up)   : [O, r, 1, 1]

    Forward:
        y = Conv_W0(x) + (alpha / r) * Conv_A(Conv_B(x))

    The implementation is self-contained and does not require PEFT.  It is
    equivalent to the convolutional low-rank parameterization described by
    FLoCoRA for standard (groups=1) ResNet convolutions.
    """

    def __init__(self, base: nn.Conv2d, rank: int, alpha: float) -> None:
        super().__init__()
        if not isinstance(base, nn.Conv2d):
            raise TypeError("base must be nn.Conv2d")
        if base.groups != 1:
            raise ValueError(
                "This FLoCoRA baseline currently supports groups=1 convolutions only."
            )
        if rank < 1:
            raise ValueError("rank must be positive")

        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

        kh, kw = base.kernel_size
        self.lora_down = nn.Conv2d(
            in_channels=base.in_channels,
            out_channels=self.rank,
            kernel_size=(kh, kw),
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=1,
            bias=False,
            padding_mode=base.padding_mode,
        )
        self.lora_up = nn.Conv2d(
            in_channels=self.rank,
            out_channels=base.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        # Zero initial effective update, matching standard LoRA behavior.
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        adapter_out = self.lora_up(self.lora_down(x))
        return base_out + adapter_out * self.scaling

    def effective_delta_weight(self) -> torch.Tensor:
        """Return the dense effective adapter update DeltaW = (alpha/r) A*B."""
        a = self.lora_up.weight[:, :, 0, 0]  # [O, r]
        b = self.lora_down.weight            # [r, I, K, K]
        return torch.einsum("or,rijk->oijk", a, b) * self.scaling

    def effective_weight(self) -> torch.Tensor:
        return self.base.weight + self.effective_delta_weight()


@dataclass(frozen=True)
class FLoCoRAStats:
    raw_dense_parameters: int
    total_parameters_with_adapters: int
    trainable_parameters: int
    adapter_parameters: int
    norm_parameters: int
    head_parameters: int
    trainable_ratio_vs_dense: float
    conv_layers_adapted: int


def _replace_child(parent: nn.Module, child_name: str, replacement: nn.Module) -> None:
    if isinstance(parent, nn.Sequential) and child_name.isdigit():
        parent[int(child_name)] = replacement
    else:
        setattr(parent, child_name, replacement)


def inject_flocora(model: nn.Module, rank: int, alpha: float) -> nn.Module:
    """
    Apply the FLoCoRA configuration used after the paper's layer ablation:
      * LoRA adapters on all Conv2d layers;
      * normalization layers remain trainable;
      * final/other Linear layers remain trainable;
      * original convolution weights stay frozen.

    The function mutates and returns ``model``.
    """
    if rank < 1:
        raise ValueError("rank must be positive")

    # Freeze everything first.  We explicitly unfreeze normalization/linear
    # layers below and the adapter parameters are trainable by construction.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def recurse(module: nn.Module) -> None:
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Conv2d):
                _replace_child(module, child_name, LoRAConv2d(child, rank, alpha))
            else:
                recurse(child)

    recurse(model)

    for module in model.modules():
        if isinstance(module, (nn.GroupNorm, nn.BatchNorm2d, nn.Linear)):
            for parameter in module.parameters(recurse=False):
                parameter.requires_grad_(True)

    return model


def trainable_named_parameters(model: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            yield name, parameter


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in trainable_named_parameters(model)
    }


@torch.no_grad()
def load_trainable_state_dict(
    model: nn.Module,
    state: dict[str, torch.Tensor],
) -> None:
    parameter_map = dict(model.named_parameters())
    expected = {name for name, _ in trainable_named_parameters(model)}
    received = set(state)
    if received != expected:
        missing = sorted(expected - received)
        extra = sorted(received - expected)
        raise ValueError(f"Trainable state mismatch; missing={missing}, extra={extra}")

    for name, value in state.items():
        parameter = parameter_map[name]
        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def count_flocora_parameters(model: nn.Module, raw_dense_parameters: int) -> FLoCoRAStats:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    adapter = 0
    norm = 0
    head = 0
    conv_layers = 0
    for module in model.modules():
        if isinstance(module, LoRAConv2d):
            conv_layers += 1
            adapter += module.lora_down.weight.numel()
            adapter += module.lora_up.weight.numel()
        elif isinstance(module, (nn.GroupNorm, nn.BatchNorm2d)):
            norm += sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        elif isinstance(module, nn.Linear):
            head += sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)

    if adapter + norm + head != trainable:
        # The current ResNet18 baseline should hit equality.  Fail loudly if
        # a future model introduces an unexpected trainable parameter.
        raise RuntimeError(
            "Unexpected trainable parameters in FLoCoRA model: "
            f"trainable={trainable}, accounted={adapter + norm + head}"
        )

    return FLoCoRAStats(
        raw_dense_parameters=int(raw_dense_parameters),
        total_parameters_with_adapters=int(total),
        trainable_parameters=int(trainable),
        adapter_parameters=int(adapter),
        norm_parameters=int(norm),
        head_parameters=int(head),
        trainable_ratio_vs_dense=float(trainable / max(raw_dense_parameters, 1)),
        conv_layers_adapted=int(conv_layers),
    )
