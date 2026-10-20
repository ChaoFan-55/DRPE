from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import get_dataset_spec
from src.models import build_model
from flocora_baseline.flocora import count_flocora_parameters, inject_flocora


def scan(target_ratio: float, min_rank: int, max_rank: int, alpha_multiplier: float):
    spec = get_dataset_spec("cifar10")
    dense = build_model("resnet18", 10, "group", spec.image_size)
    raw = sum(p.numel() for p in dense.parameters())

    rows = []
    for rank in range(min_rank, max_rank + 1):
        model = build_model("resnet18", 10, "group", spec.image_size)
        inject_flocora(model, rank=rank, alpha=alpha_multiplier * rank)
        stats = count_flocora_parameters(model, raw)
        rows.append(
            {
                "rank": rank,
                "alpha": alpha_multiplier * rank,
                "trainable_parameters": stats.trainable_parameters,
                "ratio_vs_dense": stats.trainable_ratio_vs_dense,
                "abs_error_to_target": abs(stats.trainable_ratio_vs_dense - target_ratio),
            }
        )
    rows.sort(key=lambda item: item["abs_error_to_target"])
    return raw, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate FLoCoRA rank to a target uplink ratio.")
    parser.add_argument("--target-ratio", type=float, default=0.30)
    parser.add_argument("--min-rank", type=int, default=1)
    parser.add_argument("--max-rank", type=int, default=160)
    parser.add_argument("--alpha-multiplier", type=float, default=16.0)
    parser.add_argument("--csv", type=str, default="")
    args = parser.parse_args()

    raw, rows = scan(args.target_ratio, args.min_rank, args.max_rank, args.alpha_multiplier)
    print(f"Dense ResNet18 parameters: {raw:,}")
    print(f"Target trainable/uplink ratio: {args.target_ratio:.6f}")
    print("Top candidates:")
    for row in rows[:10]:
        print(
            f"  r={row['rank']:3d} | alpha={row['alpha']:.1f} | "
            f"trainable={row['trainable_parameters']:,} | "
            f"ratio={row['ratio_vs_dense']:.6f} | "
            f"error={row['abs_error_to_target']:.6f}"
        )

    best = rows[0]
    print("-" * 88)
    print(
        f"Recommended: rank={best['rank']}, alpha={best['alpha']:.1f}, "
        f"ratio={best['ratio_vs_dense']:.6f}"
    )

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda item: item["rank"]))
        print(f"Saved: {path.resolve()}")


if __name__ == "__main__":
    main()
