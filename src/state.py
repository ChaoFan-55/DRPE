from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn as nn


def clone_named_parameters(model: nn.Module, device: torch.device | str = "cpu") -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone().to(device)
        for name, parameter in model.named_parameters()
    }


def clone_named_buffers(model: nn.Module, device: torch.device | str = "cpu") -> Dict[str, torch.Tensor]:
    return {
        name: buffer.detach().clone().to(device)
        for name, buffer in model.named_buffers()
    }


def compute_parameter_update(
    local_model: nn.Module,
    global_parameters: Mapping[str, torch.Tensor],
    device: torch.device | str = "cpu",
) -> Dict[str, torch.Tensor]:
    updates: Dict[str, torch.Tensor] = {}
    for name, parameter in local_model.named_parameters():
        updates[name] = parameter.detach().to(device).float() - global_parameters[name].to(device).float()
    return updates


def global_l2_norm(tensors: Mapping[str, torch.Tensor]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for tensor in tensors.values():
        x = tensor.detach().double().cpu()
        total += torch.sum(x * x)
    return float(torch.sqrt(total).item())


def clip_tensor_dict_by_global_l2(
    tensors: Mapping[str, torch.Tensor],
    clip_norm: float | None,
) -> tuple[Dict[str, torch.Tensor], float, float]:
    norm = global_l2_norm(tensors)
    if clip_norm is None or norm <= clip_norm:
        return {name: tensor.detach().clone() for name, tensor in tensors.items()}, norm, 1.0

    scale = clip_norm / max(norm, 1e-12)
    clipped = {name: tensor.detach().clone() * scale for name, tensor in tensors.items()}
    return clipped, norm, scale


@torch.no_grad()
def apply_parameter_update(
    model: nn.Module,
    update: Mapping[str, torch.Tensor],
    server_lr: float = 1.0,
) -> None:
    parameter_map = dict(model.named_parameters())
    missing = set(parameter_map) - set(update)
    if missing:
        raise KeyError(f"Missing updates for parameters: {sorted(missing)}")

    for name, delta in update.items():
        parameter = parameter_map[name]
        parameter.add_(delta.to(device=parameter.device, dtype=parameter.dtype), alpha=server_lr)


@torch.no_grad()
def load_aggregated_buffers(
    model: nn.Module,
    floating_buffers: Mapping[str, torch.Tensor],
    integer_buffers: Mapping[str, torch.Tensor],
) -> None:
    buffer_map = dict(model.named_buffers())
    for name, value in floating_buffers.items():
        if name in buffer_map:
            buffer_map[name].copy_(value.to(buffer_map[name].device, dtype=buffer_map[name].dtype))
    for name, value in integer_buffers.items():
        if name in buffer_map:
            buffer_map[name].copy_(value.to(buffer_map[name].device, dtype=buffer_map[name].dtype))
