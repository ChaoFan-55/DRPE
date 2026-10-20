from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .client import FederatedClient
from .config import ExperimentConfig, resolve_drpe_privacy_parameters
from .data import build_federated_data, get_dataset_spec
from .models import build_model
from .projection import ProjectionSettings
from .server import FederatedServer
from .utils import append_csv, bytes_to_mib, ensure_dir, resolve_device, save_json, set_global_seed


def learning_rate_for_round(cfg: ExperimentConfig, round_index: int) -> float:
    return cfg.learning_rate * (cfg.lr_decay_gamma ** ((round_index - 1) // cfg.lr_decay_step))


def make_projection_settings(cfg: ExperimentConfig, round_index: int):
    if cfg.method != "drpe":
        return None
    return ProjectionSettings(
        compression_ratio=cfg.compression_ratio,
        block_size=cfg.block_size,
        base_seed=cfg.projection_seed,
        round_index=round_index,
        dynamic=cfg.dynamic_projection,
        reconstruction_mode=cfg.reconstruction_mode,
        device=cfg.projection_device,
    )


def _environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def run_experiment(cfg: ExperimentConfig) -> Dict[str, float]:
    cfg.validate()
    set_global_seed(cfg.seed, deterministic=cfg.deterministic)
    device = resolve_device(cfg.device)
    output_dir = ensure_dir(cfg.output_dir)
    metrics_path = output_dir / "metrics.csv"
    if metrics_path.exists():
        metrics_path.unlink()

    spec = get_dataset_spec(cfg.dataset)
    model = build_model(cfg.model_name, cfg.num_classes, cfg.norm, spec.image_size)
    privacy = resolve_drpe_privacy_parameters(cfg, model)
    cfg.save(output_dir / "resolved_config.json")
    save_json(privacy, output_dir / "privacy_parameters.json")
    save_json(_environment(), output_dir / "environment.json")

    data = build_federated_data(cfg)
    save_json({
        "client_sizes": data.client_sizes,
        "min_client_size": min(data.client_sizes),
        "max_client_size": max(data.client_sizes),
        "mean_client_size": float(np.mean(data.client_sizes)),
        "partition_seed": cfg.partition_seed,
    }, output_dir / "partition_summary.json")

    server = FederatedServer(model=model, cfg=cfg, device=device)
    clients = [FederatedClient(i, loader, cfg, device) for i, loader in enumerate(data.client_loaders)]
    selection_rng = np.random.default_rng(cfg.seed + 999)
    selected_count = min(cfg.num_clients, max(1, int(round(cfg.num_clients * cfg.client_fraction))))

    best_val_accuracy = float("-inf")
    best_round = 0
    rounds_without_improvement = 0
    cumulative_uplink = 0
    cumulative_downlink = 0
    best_checkpoint = output_dir / "best_model.pt"
    last_checkpoint = output_dir / "last_model.pt"

    print("=" * 96)
    print(f"Dataset={cfg.dataset} | Model={cfg.model_name} | Method={cfg.method.upper()} | Device={device}")
    print(f"Clients={cfg.num_clients}, selected/round={selected_count}, alpha={cfg.dirichlet_alpha}, local epochs={cfg.local_epochs}")
    if cfg.method == "drpe":
        print(
            f"DRPE ratio={cfg.compression_ratio}, block={cfg.block_size}, dynamic={cfg.dynamic_projection}, "
            f"privacy={cfg.privacy_enabled}, clip={cfg.clip_norm}, eta={cfg.noise_eta}, sigma={cfg.noise_std}"
        )
    print("=" * 96)

    for round_index in range(1, cfg.rounds + 1):
        start = time.perf_counter()
        lr = learning_rate_for_round(cfg, round_index)
        selected_ids = selection_rng.choice(cfg.num_clients, size=selected_count, replace=False).tolist()
        projection = make_projection_settings(cfg, round_index)
        results = [clients[cid].train(server.model, lr, projection) for cid in selected_ids]
        aggregation = server.aggregate(results, projection)

        round_downlink = server.parameter_bytes() * selected_count
        cumulative_uplink += aggregation.total_uplink_bytes
        cumulative_downlink += round_downlink
        val_result = None
        if round_index % cfg.eval_every == 0:
            val_result = server.evaluate(data.val_loader)
            if val_result.accuracy > best_val_accuracy:
                best_val_accuracy = val_result.accuracy
                best_round = round_index
                rounds_without_improvement = 0
                server.save_checkpoint(str(best_checkpoint))
            else:
                rounds_without_improvement += 1

        elapsed = time.perf_counter() - start
        raw_uplink = server.parameter_bytes() * selected_count
        ratio = aggregation.total_uplink_bytes / max(raw_uplink, 1)
        append_csv({
            "round": round_index,
            "learning_rate": lr,
            "selected_clients": "|".join(map(str, selected_ids)),
            "mean_client_loss": aggregation.mean_client_loss,
            "mean_raw_update_norm": aggregation.mean_update_norm,
            "mean_clip_scale": aggregation.mean_clip_scale,
            "decoded_update_norm": aggregation.decoded_update_norm,
            "val_loss": "" if val_result is None else val_result.loss,
            "val_accuracy": "" if val_result is None else val_result.accuracy,
            "round_uplink_mib": bytes_to_mib(aggregation.total_uplink_bytes),
            "round_downlink_mib": bytes_to_mib(round_downlink),
            "uplink_ratio_vs_raw_parameters": ratio,
            "cumulative_uplink_mib": bytes_to_mib(cumulative_uplink),
            "cumulative_downlink_mib": bytes_to_mib(cumulative_downlink),
            "round_seconds": elapsed,
        }, metrics_path)

        if round_index == 1 or round_index % 5 == 0:
            val_text = "not evaluated" if val_result is None else f"{val_result.accuracy:.2f}%"
            print(
                f"Round {round_index:03d} | lr={lr:.5f} | val={val_text} | "
                f"loss={aggregation.mean_client_loss:.4f} | uplink={bytes_to_mib(aggregation.total_uplink_bytes):.2f} MiB "
                f"({ratio:.3f}x raw) | {elapsed:.1f}s"
            )
        if round_index % cfg.save_every == 0:
            server.save_checkpoint(str(output_dir / f"checkpoint_round_{round_index:04d}.pt"))
        if val_result is not None and cfg.early_stop_patience > 0 and rounds_without_improvement >= cfg.early_stop_patience:
            print(f"Early stop at round {round_index}; best={best_val_accuracy:.2f}% at {best_round}.")
            break

    server.save_checkpoint(str(last_checkpoint))
    if best_checkpoint.exists():
        server.load_checkpoint(str(best_checkpoint))
    test = server.evaluate(data.test_loader)
    summary = {
        "dataset": cfg.dataset,
        "model": cfg.model_name,
        "method": cfg.method,
        "seed": cfg.seed,
        "best_round": float(best_round),
        "best_validation_accuracy": float(best_val_accuracy),
        "test_accuracy": float(test.accuracy),
        "test_loss": float(test.loss),
        "cumulative_uplink_mib": bytes_to_mib(cumulative_uplink),
        "cumulative_downlink_mib": bytes_to_mib(cumulative_downlink),
    }
    save_json(summary, output_dir / "summary.json")
    print("-" * 96)
    print(f"Best round={best_round}, best val={best_val_accuracy:.2f}%, final test={test.accuracy:.2f}%")
    print(f"Artifacts saved to: {output_dir.resolve()}")
    return summary
