from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
TABLES = ROOT / "tables"
TABLES.mkdir(exist_ok=True)

rows = []
for path in OUTPUTS.glob("*/*/*/seed*/summary.json"):
    data = json.loads(path.read_text(encoding="utf-8"))
    parts = path.relative_to(OUTPUTS).parts
    dataset, model, method, seed_dir = parts[:4]
    data.update(dataset=dataset, model=model, method=method, seed=int(seed_dir.replace("seed", "")), output_dir=str(path.parent))
    rows.append(data)

fields = ["dataset", "model", "method", "seed", "best_round", "best_validation_accuracy", "test_accuracy", "test_loss", "cumulative_uplink_mib", "cumulative_downlink_mib", "output_dir"]
with (TABLES / "main_results_by_seed.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in sorted(rows, key=lambda x: (x["dataset"], x["model"], x["method"], x["seed"])):
        writer.writerow({key: row.get(key, "") for key in fields})

groups = defaultdict(list)
for row in rows:
    groups[(row["dataset"], row["model"], row["method"])].append(row)
summary_fields = ["dataset", "model", "method", "num_seeds", "test_accuracy_mean", "test_accuracy_std", "best_val_mean", "best_val_std", "uplink_mean_mib"]
with (TABLES / "main_results_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=summary_fields)
    writer.writeheader()
    for key, values in sorted(groups.items()):
        test = np.asarray([v["test_accuracy"] for v in values], dtype=float)
        val = np.asarray([v["best_validation_accuracy"] for v in values], dtype=float)
        uplink = np.asarray([v["cumulative_uplink_mib"] for v in values], dtype=float)
        writer.writerow({
            "dataset": key[0], "model": key[1], "method": key[2], "num_seeds": len(values),
            "test_accuracy_mean": test.mean(), "test_accuracy_std": test.std(ddof=1) if len(test)>1 else "",
            "best_val_mean": val.mean(), "best_val_std": val.std(ddof=1) if len(val)>1 else "",
            "uplink_mean_mib": uplink.mean(),
        })
print(f"Wrote summaries to {TABLES}")
