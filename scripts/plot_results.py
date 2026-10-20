from __future__ import annotations

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

for dataset_dir in [p for p in OUTPUTS.iterdir() if p.is_dir()]:
    for model_dir in [p for p in dataset_dir.iterdir() if p.is_dir()]:
        series = {}
        for method_dir in [p for p in model_dir.iterdir() if p.is_dir() and p.name in {"fedavg", "drpe"}]:
            per_seed = []
            for metrics in sorted(method_dir.glob("seed*/metrics.csv")):
                rows = list(csv.DictReader(metrics.open("r", encoding="utf-8")))
                values = [(int(r["round"]), float(r["val_accuracy"])) for r in rows if r["val_accuracy"]]
                per_seed.append(values)
            if per_seed:
                common = sorted(set.intersection(*[set(r for r, _ in seq) for seq in per_seed]))
                matrix = np.asarray([[dict(seq)[r] for r in common] for seq in per_seed])
                series[method_dir.name] = (np.asarray(common), matrix.mean(0), matrix.std(0, ddof=1) if len(matrix)>1 else np.zeros(len(common)))
        if not series:
            continue
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for method, (rounds, mean, std) in series.items():
            ax.plot(rounds, mean, label=method.upper())
            ax.fill_between(rounds, mean-std, mean+std, alpha=0.18)
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Validation accuracy (%)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        stem = FIGURES / f"convergence_{dataset_dir.name}_{model_dir.name}"
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
        plt.close(fig)
print(f"Figures saved to {FIGURES}")
