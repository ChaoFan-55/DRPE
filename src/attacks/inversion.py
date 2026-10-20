from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .operators import select_named_parameters


def normalize_image(image_01: torch.Tensor, mean, std) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=image_01.device, dtype=image_01.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=image_01.device, dtype=image_01.dtype).view(1, 3, 1, 1)
    return (image_01 - mean_t) / std_t


def total_variation(image: torch.Tensor) -> torch.Tensor:
    horizontal = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    vertical = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    return horizontal + vertical


def compute_named_gradients(
    model: nn.Module,
    image_01: torch.Tensor,
    label: torch.Tensor | None,
    parameter_scope: str,
    create_graph: bool,
    mean,
    std,
    label_logits: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    selected = select_named_parameters(model, parameter_scope)
    names = [name for name, _ in selected]
    parameters = [parameter for _, parameter in selected]
    logits = model(normalize_image(image_01, mean, std))
    if label_logits is None:
        if label is None:
            raise ValueError("A hard label is required when label_logits is None.")
        loss = F.cross_entropy(logits, label)
    else:
        soft_label = torch.softmax(label_logits, dim=0).unsqueeze(0)
        loss = -(soft_label * F.log_softmax(logits, dim=1)).sum()
    gradients = torch.autograd.grad(
        loss,
        parameters,
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=False,
    )
    return {name: gradient for name, gradient in zip(names, gradients)}
