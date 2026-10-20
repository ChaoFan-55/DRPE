from __future__ import annotations

import json
import sys
from pathlib import Path

from .common import BENCH_DIR, PROJECT_ROOT, load_methods, load_protocol, root_path


def output_dir_for(row: dict) -> Path:
    return (
        BENCH_DIR / "results" / "raw" / row["method"] /
        f"seed{int(row['model_seed'])}" / row["target_id"] /
        f"restart{int(row['restart']):02d}" / row["attack"]
    )


def _classic_config(row: dict, method_cfg: dict, output_dir: Path) -> Path:
    p = load_protocol()
    model_seed = int(row["model_seed"])
    attack = row["attack"]
    observation_mode = method_cfg["observation_by_attack"][attack]
    checkpoint = method_cfg["checkpoint_template"].format(model_seed=model_seed)
    training_summary = method_cfg.get("training_summary_template", "").format(model_seed=model_seed)

    cfg = {
        "seed": int(row["attack_base_seed"]),
        "dataset": p["dataset"],
        "data_root": p["data_root"],
        "dataset_download": True,
        "device": "auto",
        "model_name": p["model"],
        "num_classes": int(p["num_classes"]),
        "norm": "group",
        "target_split": p["target_split"],
        "sample_indices": [int(row["sample_index"])],
        "parameter_scope": "all",
        "iterations": int(p["iterations"]),
        "attack_lr": float(p["attack_lr"]),
        "optimizer": p["optimizer"],
        "matching_loss": p["matching_loss"],
        "tv_weight": float(p["tv_weight"]),
        "image_l2_weight": 0.0,
        "label_entropy_weight": float(p["label_entropy_weight"]),
        "init": "uniform",
        "image_parameterization": "sigmoid",
        "label_init": "zeros",
        "label_lr": float(p["label_lr"]),
        "attack_grad_clip_norm": None,
        "relative_l2_floor": 1e-8,
        "save_every": int(p["save_every"]),
        "print_every": int(p["print_every"]),
        "paired_initialization": True,
        "lbfgs_history_size": 100,
        "lbfgs_max_iter_per_step": 1,
        "save_observations": False,
        "compute_optional_lpips": bool(p["compute_lpips"]),
        "compression_ratio": 0.3,
        "block_size": 512,
        "dynamic_projection": True,
        "projection_seed": 2026,
        "projection_round": int(p["transmission_round"]),
        "reconstruction_mode": "projected",
        "projection_device": "cpu",
        "projection_batch_blocks": 256,
        "noise_seed": 314159,
        "observed_client_id": int(p["observed_client_id"]),
        "noise_eta": 2.0,
        "output_dir": str(output_dir),
        "checkpoint_path": checkpoint,
        "training_summary_path": training_summary or None,
        "attack_name": attack,
        "label_mode": "optimize",
        "observation_mode": observation_mode,
        "observation_clip_norm": None,
        "calibration_path": None,
        "noise_std": 0.0,
    }

    if row["method"] == "drpe":
        cfg.update({
            "projection_seed": int(method_cfg["projection_seed"][str(model_seed)]),
            "noise_seed": int(method_cfg["noise_seed"][str(model_seed)]),
            "noise_eta": float(method_cfg["extra"]["noise_eta"]),
            "calibration_path": method_cfg["calibration_path"],
            "noise_std": None,
            "compression_ratio": float(method_cfg["extra"]["compression_ratio"]),
            "block_size": int(method_cfg["extra"]["block_size"]),
            "dynamic_projection": bool(method_cfg["extra"]["dynamic_projection"]),
        })
    else:
        # FedAvg is raw and has no clipping/noise.
        cfg.update({
            "noise_eta": 0.0,
            "noise_std": 0.0,
            "calibration_path": None,
            "observation_clip_norm": None,
        })

    config_dir = BENCH_DIR / "results" / "generated_configs" / row["method"]
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{row['run_id']}.json"
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return config_path


def build_command(row: dict) -> tuple[list[str], Path]:
    protocol = load_protocol()
    methods = load_methods()
    method = row["method"]
    if method not in methods:
        raise KeyError(f"Unknown method: {method}")
    m = methods[method]
    output_dir = output_dir_for(row)
    output_dir.mkdir(parents=True, exist_ok=True)

    if m["kind"] == "classic":
        config_path = _classic_config(row, m, output_dir)
        cmd = [sys.executable, str(root_path(m["script"])), "--config", str(config_path)]
        return cmd, output_dir

    model_seed = int(row["model_seed"])
    checkpoint = m["checkpoint_template"].format(model_seed=model_seed)
    cmd = [
        sys.executable, str(root_path(m["script"])),
        "--checkpoint", checkpoint,
        "--output-dir", str(output_dir),
        "--attack", row["attack"],
        # This is intentionally the attack base seed, not the training/model seed.
        "--seed", str(int(row["attack_base_seed"])),
        "--round-index", str(int(row["transmission_round"])),
        "--sample-index", str(int(row["sample_index"])),
        "--iterations", str(int(protocol["iterations"])),
        "--attack-lr", str(protocol["attack_lr"]),
        "--label-lr", str(protocol["label_lr"]),
        "--tv-weight", str(protocol["tv_weight"]),
        "--label-entropy-weight", str(protocol["label_entropy_weight"]),
        "--print-every", str(int(protocol["print_every"])),
        "--data-root", protocol["data_root"],
    ]

    if not protocol.get("compute_lpips", True):
        cmd.append("--no-lpips")

    extra = m.get("extra", {})
    if method == "random_k":
        cmd += [
            "--random-k-seed", str(m["method_seed"][str(model_seed)]),
            "--client-id", str(protocol["observed_client_id"]),
            "--keep-ratio", str(extra["keep_ratio"]),
        ]
    elif method == "top_k":
        cmd += [
            "--client-id", str(protocol["observed_client_id"]),
            "--keep-ratio", str(extra["keep_ratio"]),
        ]
    elif method == "dense_rp":
        cmd += [
            "--projection-seed", str(m["method_seed"][str(model_seed)]),
            "--compression-ratio", str(extra["compression_ratio"]),
            "--block-size", str(extra["block_size"]),
            "--projection-batch-blocks", str(extra["projection_batch_blocks"]),
        ]
    elif method == "powersgd":
        cmd += [
            "--powersgd-seed", str(m["method_seed"][str(model_seed)]),
            "--client-id", str(protocol["observed_client_id"]),
            "--rank", str(extra["rank"]),
            "--power-iterations", str(extra["power_iterations"]),
        ]
    elif method == "flocora":
        cmd += [
            "--observation", str(extra["observation"]),
            "--rank", str(extra["rank"]),
            "--lora-alpha", str(extra["lora_alpha"]),
            "--virtual-local-lr", str(extra["virtual_local_lr"]),
        ]
    else:
        raise ValueError(f"No CLI adapter implemented for method={method}")
    return cmd, output_dir


def checkpoint_for(method: str, model_seed: int) -> Path:
    m = load_methods()[method]
    return root_path(m["checkpoint_template"].format(model_seed=model_seed))


def script_for(method: str) -> Path:
    return root_path(load_methods()[method]["script"])
