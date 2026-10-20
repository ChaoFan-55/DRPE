from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .compression import encode_update_dict
from .config import ExperimentConfig
from .projection import ProjectionSettings
from .state import (
    clip_tensor_dict_by_global_l2,
    clone_named_buffers,
    clone_named_parameters,
    compute_parameter_update,
)


@dataclass
class ClientResult:
    client_id: int
    num_samples: int
    method: str
    payload: Dict[str, torch.Tensor]
    original_shapes: Dict[str, torch.Size]
    floating_buffers: Dict[str, torch.Tensor]
    integer_buffers: Dict[str, torch.Tensor]
    local_loss: float
    update_norm: float
    clip_scale: float
    uplink_bytes: int


class FederatedClient:
    def __init__(self, client_id: int, dataloader: DataLoader, cfg: ExperimentConfig, device: torch.device):
        self.client_id = client_id
        self.dataloader = dataloader
        self.cfg = cfg
        self.device = device
        self.num_samples = len(dataloader.dataset)

    def _train_local_model(self, local_model: nn.Module, learning_rate: float) -> float:
        local_model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=learning_rate,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay,
        )

        total_loss = 0.0
        total_examples = 0
        for _ in range(self.cfg.local_epochs):
            for inputs, targets in self.dataloader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = local_model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                batch_size = inputs.size(0)
                total_loss += float(loss.item()) * batch_size
                total_examples += batch_size

        return total_loss / max(total_examples, 1)

    def train(
        self,
        global_model: nn.Module,
        learning_rate: float,
        projection_settings: ProjectionSettings | None,
    ) -> ClientResult:
        # A real clone is mandatory. The client must never mutate the server model.
        global_parameters = clone_named_parameters(global_model, device="cpu")
        local_model = copy.deepcopy(global_model).to(self.device)

        local_loss = self._train_local_model(local_model, learning_rate)
        raw_updates = compute_parameter_update(
            local_model=local_model,
            global_parameters=global_parameters,
            device="cpu",
        )
        clipped_updates, update_norm, clip_scale = clip_tensor_dict_by_global_l2(
            raw_updates,
            clip_norm=self.cfg.clip_norm,
        )

        local_buffers = clone_named_buffers(local_model, device="cpu")
        floating_buffers = {
            name: tensor.float()
            for name, tensor in local_buffers.items()
            if torch.is_floating_point(tensor)
        }
        integer_buffers = {
            name: tensor
            for name, tensor in local_buffers.items()
            if not torch.is_floating_point(tensor)
        }
        buffer_bytes = sum(t.numel() * t.element_size() for t in local_buffers.values())

        if self.cfg.method == "fedavg":
            payload = {name: tensor.detach().clone().float() for name, tensor in clipped_updates.items()}
            original_shapes = {name: tensor.shape for name, tensor in clipped_updates.items()}
            payload_bytes = sum(t.numel() * t.element_size() for t in payload.values())
        else:
            if projection_settings is None:
                raise ValueError("DRPE requires projection settings.")
            encoded = encode_update_dict(
                updates=clipped_updates,
                settings=projection_settings,
                noise_std=float(self.cfg.noise_std or 0.0),
                client_id=self.client_id,
                noise_seed=self.cfg.noise_seed,
            )
            payload = encoded.tensors
            original_shapes = encoded.original_shapes
            payload_bytes = encoded.payload_bytes

        # Explicitly drop the client copy before returning to reduce peak memory.
        del local_model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return ClientResult(
            client_id=self.client_id,
            num_samples=self.num_samples,
            method=self.cfg.method,
            payload=payload,
            original_shapes=original_shapes,
            floating_buffers=floating_buffers,
            integer_buffers=integer_buffers,
            local_loss=local_loss,
            update_norm=update_norm,
            clip_scale=clip_scale,
            uplink_bytes=payload_bytes + buffer_bytes,
        )
