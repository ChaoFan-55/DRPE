from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from torchvision.datasets import CIFAR10
from runners.common import load_protocol, root_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one deterministic CIFAR-10 train target per class.")
    parser.add_argument("--output", default=str(BENCH / "target_samples.csv"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    p = load_protocol()
    if p["dataset"].lower() != "cifar10":
        raise ValueError("This v1 target builder currently supports CIFAR-10 only.")
    if p["target_split"] != "train":
        raise ValueError("This benchmark is currently fixed to the CIFAR-10 train attack split.")

    output = Path(args.output)
    if output.exists() and output.stat().st_size > 60 and not args.overwrite:
        print(f"Target file already populated: {output}")
        print("Use --overwrite only when you intentionally want to regenerate the fixed target set.")
        return

    dataset = CIFAR10(root=str(root_path(p["data_root"])), train=True, download=True)
    by_class: dict[int, list[int]] = {i: [] for i in range(10)}
    for idx, label in enumerate(dataset.targets):
        by_class[int(label)].append(idx)

    rows = []
    base_seed = int(p["target_selection_seed"])
    for class_id in range(10):
        rng = random.Random(base_seed + class_id * 10007)
        dataset_index = rng.choice(by_class[class_id])
        rows.append({
            "target_id": f"T{class_id:02d}",
            "dataset_index": dataset_index,
            "class_id": class_id,
            "class_name": dataset.classes[class_id],
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["target_id", "dataset_index", "class_id", "class_name"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} fixed class-balanced targets to: {output}")
    for row in rows:
        print(f"  {row['target_id']} class={row['class_id']} {row['class_name']:<10s} index={row['dataset_index']}")


if __name__ == "__main__":
    main()
