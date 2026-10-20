from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, model_encoded_numel
from src.data import get_dataset_spec
from src.models import build_model
from src.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run a no-privacy DRPE calibration and compute P75.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.method != "drpe" or cfg.privacy_enabled:
        raise ValueError("Calibration config must use method='drpe' and privacy_enabled=false.")
    run_experiment(cfg)
    output = Path(cfg.output_dir)
    rows = list(csv.DictReader((output / "metrics.csv").open("r", encoding="utf-8")))
    norms = np.asarray([float(row["mean_raw_update_norm"]) for row in rows], dtype=np.float64)
    spec = get_dataset_spec(cfg.dataset)
    model = build_model(cfg.model_name, cfg.num_classes, cfg.norm, spec.image_size)
    encoded_count = model_encoded_numel(model, cfg.block_size, cfg.compression_ratio)
    p50, p75, p90 = [float(np.quantile(norms, q)) for q in (0.50, 0.75, 0.90)]
    eta = 2.0
    result = {
        "dataset": cfg.dataset,
        "model": cfg.model_name,
        "calibration_seed": cfg.seed,
        "source_rounds": len(norms),
        "statistic": "quantile of round-mean client raw update norms",
        "p50": p50,
        "p75": p75,
        "p90": p90,
        "selected_clip_norm": p75,
        "compression_ratio": cfg.compression_ratio,
        "block_size": cfg.block_size,
        "encoded_parameter_count": int(encoded_count),
        "noise_eta": eta,
        "derived_noise_std": float(eta * p75 / np.sqrt(encoded_count)),
    }
    (output / "calibration.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
