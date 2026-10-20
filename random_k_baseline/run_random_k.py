from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# Existing verified project components
# ============================================================

from src.client import FederatedClient
from src.config import ExperimentConfig
from src.data import (
    build_federated_data,
    get_dataset_spec,
)
from src.models import build_model
from src.server import FederatedServer
from src.utils import (
    append_csv,
    bytes_to_mib,
    ensure_dir,
    resolve_device,
    save_json,
    set_global_seed,
)

from random_k import (
    aggregate_random_k,
    compress_client_result,
)


# ============================================================
# Config utilities
# ============================================================

def parse_scalar(text: str) -> Any:

    lower = text.lower()

    if lower == "true":
        return True

    if lower == "false":
        return False

    if lower in {
        "none",
        "null",
    }:
        return None

    try:

        if any(
            symbol in text
            for symbol in [
                ".",
                "e",
                "E",
            ]
        ):
            return float(text)

        return int(text)

    except ValueError:
        return text


def load_settings(
    path: str | Path,
    overrides: list[str],
) -> Dict[str, Any]:

    settings = json.loads(
        Path(path).read_text(
            encoding="utf-8",
        )
    )

    for item in overrides:

        if "=" not in item:
            raise ValueError(
                f"Invalid override: {item}"
            )

        key, value = item.split(
            "=",
            1,
        )

        settings[key.strip()] = (
            parse_scalar(
                value.strip()
            )
        )

    if (
        settings.get("method")
        != "random_k"
    ):
        raise ValueError(
            "method must be 'random_k'."
        )

    compression_ratio = float(
        settings.get(
            "compression_ratio",
            0.15,
        )
    )

    if not (
        0 < compression_ratio <= 1
    ):
        raise ValueError(
            "compression_ratio must be in (0, 1]."
        )

    return settings


def make_base_config(
    settings: Dict[str, Any],
) -> ExperimentConfig:
    """
    Reuse the existing FedAvg client solely for:

    - dataset configuration
    - local training
    - dense model update computation
    - buffer handling

    Random-k compression and aggregation remain outside src/.
    """

    valid_fields = {
        field.name
        for field
        in fields(ExperimentConfig)
    }

    config_data = {
        key: value
        for key, value
        in settings.items()
        if key in valid_fields
    }

    # Existing FederatedClient must use its FedAvg/raw-update path.
    config_data["method"] = "fedavg"

    # Random-k baseline has no privacy mechanism.
    config_data["privacy_enabled"] = False
    config_data["clip_norm"] = None
    config_data["noise_eta"] = 0.0
    config_data["noise_std"] = 0.0

    # No DRPE compression inside the existing client.
    config_data["compression_ratio"] = 1.0

    cfg = ExperimentConfig(
        **config_data
    )

    cfg.validate()

    return cfg


# ============================================================
# Learning-rate schedule
# ============================================================

def learning_rate_for_round(
    cfg: ExperimentConfig,
    round_index: int,
) -> float:

    decay_count = (
        (round_index - 1)
        // cfg.lr_decay_step
    )

    return (
        cfg.learning_rate
        * (
            cfg.lr_decay_gamma
            ** decay_count
        )
    )


# ============================================================
# Environment information
# ============================================================

def environment_info() -> dict:

    return {
        "python":
            platform.python_version(),

        "torch":
            torch.__version__,

        "cuda_runtime":
            torch.version.cuda,

        "cuda_available":
            torch.cuda.is_available(),

        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),

        "CUBLAS_WORKSPACE_CONFIG":
            os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
    }


# ============================================================
# Evaluation helper
# ============================================================

def evaluate_with_temporary_gpu(
    server: FederatedServer,
    dataloader,
    device: torch.device,
):
    """
    Server/global model normally stays on CPU.

    For validation/test only:
        CPU -> GPU -> evaluate -> CPU

    This prevents the global model from occupying GPU memory
    while the local client model is being trained.
    """

    server.model.to(device)

    try:
        result = server.evaluate(
            dataloader
        )

    finally:
        server.model.cpu()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return result


# ============================================================
# Main experiment
# ============================================================

def run(
    settings: Dict[str, Any],
) -> Dict[str, Any]:

    # --------------------------------------------------------
    # Build base FedAvg configuration for shared infrastructure
    # --------------------------------------------------------

    cfg = make_base_config(
        settings
    )

    compression_ratio = float(
        settings.get(
            "compression_ratio",
            0.15,
        )
    )

    random_k_seed = int(
        settings.get(
            "random_k_seed",
            2026,
        )
    )

    unbiased = bool(
        settings.get(
            "unbiased",
            True,
        )
    )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_global_seed(
        cfg.seed,
        deterministic=cfg.deterministic,
    )

    device = resolve_device(
        cfg.device
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir = ensure_dir(
        cfg.output_dir
    )

    metrics_path = (
        output_dir
        / "metrics.csv"
    )

    if metrics_path.exists():
        metrics_path.unlink()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    dataset_spec = get_dataset_spec(
        cfg.dataset
    )

    model = build_model(
        cfg.model_name,
        cfg.num_classes,
        cfg.norm,
        dataset_spec.image_size,
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    data = build_federated_data(
        cfg
    )

    # --------------------------------------------------------
    # Server
    #
    # FederatedServer initially moves the model to cfg.device.
    # Immediately move the global model back to CPU.
    #
    # GPU will therefore be reserved for the current local
    # client's temporary training model.
    # --------------------------------------------------------

    server = FederatedServer(
        model=model,
        cfg=cfg,
        device=device,
    )

    server.model.cpu()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Existing verified FL clients
    # --------------------------------------------------------

    clients = [
        FederatedClient(
            client_id,
            loader,
            cfg,
            device,
        )
        for client_id, loader
        in enumerate(
            data.client_loaders
        )
    ]

    # --------------------------------------------------------
    # Client selection
    # --------------------------------------------------------

    selected_count = min(
        cfg.num_clients,
        max(
            1,
            int(
                round(
                    cfg.num_clients
                    * cfg.client_fraction
                )
            ),
        ),
    )

    selection_rng = (
        np.random.default_rng(
            cfg.seed + 999
        )
    )

    # --------------------------------------------------------
    # Save experiment information
    # --------------------------------------------------------

    save_json(
        settings,
        output_dir
        / "resolved_config.json",
    )

    save_json(
        environment_info(),
        output_dir
        / "environment.json",
    )

    save_json(
        {
            "method":
                "random_k",

            "compression_ratio":
                compression_ratio,

            "unbiased":
                unbiased,

            "random_k_seed":
                random_k_seed,

            "sampling":
                "without_replacement",

            "sampling_scope":
                "independent_per_round_client_tensor",

            "value_dtype":
                "float32",

            "index_dtype":
                "int32",

            "index_bytes_counted":
                True,

            "error_feedback":
                False,

            "privacy_noise":
                False,

            "clipping":
                False,

            "server_model_device_during_training":
                "cpu",

            "client_model_device":
                str(device),

            "expected_communication_ratio":
                2.0
                * compression_ratio,
        },
        output_dir
        / "method_metadata.json",
    )

    save_json(
        {
            "client_sizes":
                data.client_sizes,

            "min_client_size":
                min(data.client_sizes),

            "max_client_size":
                max(data.client_sizes),

            "mean_client_size":
                float(
                    np.mean(
                        data.client_sizes
                    )
                ),

            "partition_seed":
                cfg.partition_seed,
        },
        output_dir
        / "partition_summary.json",
    )

    # --------------------------------------------------------
    # Running statistics
    # --------------------------------------------------------

    best_val_accuracy = float(
        "-inf"
    )

    best_round = 0

    rounds_without_improvement = 0

    cumulative_uplink = 0
    cumulative_downlink = 0

    best_checkpoint = (
        output_dir
        / "best_model.pt"
    )

    last_checkpoint = (
        output_dir
        / "last_model.pt"
    )

    # --------------------------------------------------------
    # Console header
    # --------------------------------------------------------

    print("=" * 104)

    print(
        f"Dataset={cfg.dataset} | "
        f"Model={cfg.model_name} | "
        f"Method=RANDOM-K | "
        f"Device={device}"
    )

    print(
        f"Clients={cfg.num_clients}, "
        f"selected/round={selected_count}, "
        f"alpha={cfg.dirichlet_alpha}, "
        f"local epochs={cfg.local_epochs}, "
        f"batch={cfg.batch_size}"
    )

    print(
        f"Keep ratio={compression_ratio}, "
        f"unbiased={unbiased}, "
        f"value=float32, "
        f"index=int32, "
        f"expected≈"
        f"{2 * compression_ratio:.3f}x raw"
    )

    print(
        "Server model=CPU during client training | "
        f"Client training={device}"
    )

    print("=" * 104)

    # ========================================================
    # Federated training
    # ========================================================

    for round_index in range(
        1,
        cfg.rounds + 1,
    ):

        round_start = (
            time.perf_counter()
        )

        learning_rate = (
            learning_rate_for_round(
                cfg,
                round_index,
            )
        )

        selected_ids = (
            selection_rng.choice(
                cfg.num_clients,
                size=selected_count,
                replace=False,
            ).tolist()
        )

        # ----------------------------------------------------
        # Process clients sequentially.
        #
        # Dense update exists only temporarily:
        #
        # client training
        #      ↓
        # dense update
        #      ↓
        # Random-k immediately
        #      ↓
        # dense update released
        # ----------------------------------------------------

        sparse_results = []

        for client_id in selected_ids:

            # Global/server model is on CPU here.
            #
            # Existing FederatedClient internally deep-copies
            # it and moves only the local model to CUDA.
            dense_result = (
                clients[
                    client_id
                ].train(
                    global_model=server.model,
                    learning_rate=learning_rate,
                    projection_settings=None,
                )
            )

            # Existing client deletes the local CUDA model
            # before returning.

            sparse_result = (
                compress_client_result(
                    result=dense_result,

                    ratio=(
                        compression_ratio
                    ),

                    base_seed=(
                        random_k_seed
                    ),

                    round_index=(
                        round_index
                    ),

                    unbiased=(
                        unbiased
                    ),

                    sampling_device=(
                        device
                    ),
                )
            )

            sparse_results.append(
                sparse_result
            )

            del dense_result

            if device.type == "cuda":
                torch.cuda.empty_cache()

        # ----------------------------------------------------
        # Random-k aggregation on CPU
        # ----------------------------------------------------

        aggregation = (
            aggregate_random_k(
                model=server.model,
                results=sparse_results,
                server_lr=cfg.server_lr,
            )
        )

        del sparse_results

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # ----------------------------------------------------
        # Communication
        # ----------------------------------------------------

        raw_uplink = (
            server.parameter_bytes()
            * selected_count
        )

        round_downlink = (
            server.parameter_bytes()
            * selected_count
        )

        cumulative_uplink += (
            aggregation
            .total_uplink_bytes
        )

        cumulative_downlink += (
            round_downlink
        )

        uplink_ratio = (
            aggregation
            .total_uplink_bytes
            / max(
                raw_uplink,
                1,
            )
        )

        # ----------------------------------------------------
        # Validation
        #
        # Global model:
        # CPU -> GPU -> validation -> CPU
        # ----------------------------------------------------

        val_result = None

        if (
            round_index
            % cfg.eval_every
            == 0
        ):

            val_result = (
                evaluate_with_temporary_gpu(
                    server=server,
                    dataloader=data.val_loader,
                    device=device,
                )
            )

            if (
                val_result.accuracy
                > best_val_accuracy
            ):

                best_val_accuracy = (
                    val_result.accuracy
                )

                best_round = (
                    round_index
                )

                rounds_without_improvement = 0

                # Model is back on CPU here.
                server.save_checkpoint(
                    str(
                        best_checkpoint
                    )
                )

            else:

                rounds_without_improvement += 1

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        elapsed = (
            time.perf_counter()
            - round_start
        )

        append_csv(
            {
                "round":
                    round_index,

                "learning_rate":
                    learning_rate,

                "selected_clients":
                    "|".join(
                        map(
                            str,
                            selected_ids,
                        )
                    ),

                "mean_client_loss":
                    aggregation
                    .mean_client_loss,

                "mean_raw_update_norm":
                    aggregation
                    .mean_update_norm,

                "decoded_update_norm":
                    aggregation
                    .decoded_update_norm,

                "configured_keep_ratio":
                    compression_ratio,

                "actual_selected_fraction":
                    aggregation
                    .selected_fraction,

                "val_loss": (
                    ""
                    if val_result is None
                    else
                    val_result.loss
                ),

                "val_accuracy": (
                    ""
                    if val_result is None
                    else
                    val_result.accuracy
                ),

                "round_uplink_mib":
                    bytes_to_mib(
                        aggregation
                        .total_uplink_bytes
                    ),

                "round_downlink_mib":
                    bytes_to_mib(
                        round_downlink
                    ),

                "uplink_ratio_vs_raw_parameters":
                    uplink_ratio,

                "cumulative_uplink_mib":
                    bytes_to_mib(
                        cumulative_uplink
                    ),

                "cumulative_downlink_mib":
                    bytes_to_mib(
                        cumulative_downlink
                    ),

                "round_seconds":
                    elapsed,
            },
            metrics_path,
        )

        # ----------------------------------------------------
        # Console
        # ----------------------------------------------------

        if (
            round_index == 1
            or round_index % 5 == 0
        ):

            val_text = (
                "not evaluated"
                if val_result is None
                else
                f"{val_result.accuracy:.2f}%"
            )

            print(
                f"Round {round_index:03d} | "
                f"lr={learning_rate:.5f} | "
                f"val={val_text} | "
                f"loss="
                f"{aggregation.mean_client_loss:.4f} | "
                f"selected="
                f"{aggregation.selected_fraction:.4f} | "
                f"uplink="
                f"{bytes_to_mib(aggregation.total_uplink_bytes):.2f} MiB "
                f"({uplink_ratio:.3f}x raw) | "
                f"{elapsed:.1f}s"
            )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        if (
            round_index
            % cfg.save_every
            == 0
        ):

            # Server model is already on CPU.
            server.save_checkpoint(
                str(
                    output_dir
                    / (
                        "checkpoint_round_"
                        f"{round_index:04d}.pt"
                    )
                )
            )

        # ----------------------------------------------------
        # Early stop
        # ----------------------------------------------------

        if (
            val_result is not None
            and cfg.early_stop_patience > 0
            and rounds_without_improvement
            >= cfg.early_stop_patience
        ):

            print(
                f"Early stop at round "
                f"{round_index}; "
                f"best="
                f"{best_val_accuracy:.2f}% "
                f"at round "
                f"{best_round}."
            )

            break

    # ========================================================
    # Save last model
    # ========================================================

    server.model.cpu()

    server.save_checkpoint(
        str(
            last_checkpoint
        )
    )

    # ========================================================
    # Load best model explicitly on CPU
    # ========================================================

    if best_checkpoint.exists():

        state_dict = torch.load(
            best_checkpoint,
            map_location="cpu",
        )

        server.model.load_state_dict(
            state_dict
        )

        del state_dict

    # ========================================================
    # Final test
    #
    # CPU -> GPU -> test -> CPU
    # ========================================================

    test_result = (
        evaluate_with_temporary_gpu(
            server=server,
            dataloader=data.test_loader,
            device=device,
        )
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "dataset":
            cfg.dataset,

        "model":
            cfg.model_name,

        "method":
            "random_k",

        "seed":
            cfg.seed,

        "partition_seed":
            cfg.partition_seed,

        "random_k_seed":
            random_k_seed,

        "compression_ratio":
            compression_ratio,

        "unbiased":
            unbiased,

        "best_round":
            best_round,

        "best_validation_accuracy":
            float(
                best_val_accuracy
            ),

        "test_accuracy":
            float(
                test_result.accuracy
            ),

        "test_loss":
            float(
                test_result.loss
            ),

        "cumulative_uplink_mib":
            bytes_to_mib(
                cumulative_uplink
            ),

        "cumulative_downlink_mib":
            bytes_to_mib(
                cumulative_downlink
            ),
    }

    save_json(
        summary,
        output_dir
        / "summary.json",
    )

    print("-" * 104)

    print(
        f"Best round={best_round}, "
        f"best val="
        f"{best_val_accuracy:.2f}%, "
        f"test="
        f"{test_result.accuracy:.2f}%"
    )

    print(
        "Artifacts saved to: "
        f"{output_dir.resolve()}"
    )

    return summary


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Standalone Random-k baseline"
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to Random-k JSON config.",
    )

    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        help="Optional key=value overrides.",
    )

    args = parser.parse_args()

    settings = load_settings(
        args.config,
        args.set,
    )

    run(
        settings
    )


if __name__ == "__main__":
    main()