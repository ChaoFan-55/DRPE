from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .client import ClientResult
from .compression import decode_update_dict
from .config import ExperimentConfig
from .projection import ProjectionSettings
from .state import apply_parameter_update, load_aggregated_buffers


@dataclass
class EvaluationResult:
    loss: float
    accuracy: float


@dataclass
class AggregationResult:
    decoded_update_norm: float
    total_uplink_bytes: int
    mean_client_loss: float
    mean_update_norm: float
    mean_clip_scale: float


class FederatedServer:
    def __init__(self, model: nn.Module, cfg: ExperimentConfig, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    @staticmethod
    def _weighted_average_payloads(results: List[ClientResult]) -> Dict[str, torch.Tensor]:
        if not results:
            raise ValueError("Cannot aggregate an empty client result list.")

        total_samples = sum(result.num_samples for result in results)
        names = list(results[0].payload.keys())
        averaged: Dict[str, torch.Tensor] = {}

        for name in names:
            reference = results[0].payload[name]
            accumulator = torch.zeros_like(reference, dtype=torch.float32, device="cpu")
            for result in results:
                if name not in result.payload:
                    raise KeyError(f"Client {result.client_id} is missing payload tensor {name}.")
                weight = result.num_samples / total_samples
                accumulator.add_(result.payload[name].float().cpu(), alpha=weight)
            averaged[name] = accumulator

        return averaged

    @staticmethod
    def _aggregate_buffers(results: List[ClientResult]) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        total_samples = sum(result.num_samples for result in results)
        floating: Dict[str, torch.Tensor] = {}
        integer: Dict[str, torch.Tensor] = {}

        if results[0].floating_buffers:
            for name in results[0].floating_buffers:
                reference = results[0].floating_buffers[name]
                accumulator = torch.zeros_like(reference, dtype=torch.float32)
                for result in results:
                    weight = result.num_samples / total_samples
                    accumulator.add_(result.floating_buffers[name].float(), alpha=weight)
                floating[name] = accumulator

        if results[0].integer_buffers:
            for name in results[0].integer_buffers:
                stacked = torch.stack([result.integer_buffers[name].long() for result in results])
                integer[name] = torch.max(stacked, dim=0).values

        return floating, integer

    def aggregate(
        self,
        results: List[ClientResult],
        projection_settings: ProjectionSettings | None,
    ) -> AggregationResult:
        averaged_payload = self._weighted_average_payloads(results)

        if self.cfg.method == "fedavg":
            decoded_update = averaged_payload
        else:
            if projection_settings is None:
                raise ValueError("DRPE aggregation requires projection settings.")
            decoded_update = decode_update_dict(
                encoded=averaged_payload,
                original_shapes=results[0].original_shapes,
                settings=projection_settings,
            )

        apply_parameter_update(
            model=self.model,
            update=decoded_update,
            server_lr=self.cfg.server_lr,
        )

        floating_buffers, integer_buffers = self._aggregate_buffers(results)
        load_aggregated_buffers(self.model, floating_buffers, integer_buffers)

        update_norm_sq = 0.0
        for tensor in decoded_update.values():
            x = tensor.detach().double().cpu()
            update_norm_sq += float(torch.sum(x * x).item())

        return AggregationResult(
            decoded_update_norm=update_norm_sq**0.5,
            total_uplink_bytes=sum(result.uplink_bytes for result in results),
            mean_client_loss=sum(result.local_loss for result in results) / len(results),
            mean_update_norm=sum(result.update_norm for result in results) / len(results),
            mean_clip_scale=sum(result.clip_scale for result in results) / len(results),
        )

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> EvaluationResult:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)

            total_loss += float(loss.item()) * inputs.size(0)
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_examples += inputs.size(0)

        return EvaluationResult(
            loss=total_loss / max(total_examples, 1),
            accuracy=100.0 * total_correct / max(total_examples, 1),
        )

    def parameter_bytes(self) -> int:
        return sum(parameter.numel() * parameter.element_size() for parameter in self.model.parameters())

    def save_checkpoint(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
