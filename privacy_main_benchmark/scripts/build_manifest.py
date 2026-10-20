from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = Path(__file__).resolve().parents[1]
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from runners.common import load_protocol


def read_targets(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 10:
        raise RuntimeError(f"target_samples.csv is not populated (found {len(rows)} targets). Run build_target_samples.py first.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "main"], required=True)
    parser.add_argument("--targets", default=str(BENCH / "target_samples.csv"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    p = load_protocol()
    targets = read_targets(Path(args.targets))
    mode_cfg = p[args.mode]
    targets = targets[: int(mode_cfg["num_target_samples"])]
    attack_base_seeds = [int(x) for x in mode_cfg["attack_base_seeds"]]

    rows = []
    for method in p["methods"]:
        for model_seed in p["model_seeds"]:
            for target in targets:
                for attack in p["attacks"]:
                    for restart, attack_base_seed in enumerate(attack_base_seeds):
                        run_id = (
                            f"{method}__m{model_seed}__{target['target_id']}__"
                            f"{attack}__r{restart:02d}"
                        )
                        rows.append({
                            "run_id": run_id,
                            "method": method,
                            "model_seed": model_seed,
                            "target_id": target["target_id"],
                            "sample_index": target["dataset_index"],
                            "class_id": target["class_id"],
                            "class_name": target["class_name"],
                            "attack": attack,
                            "restart": restart,
                            "attack_base_seed": attack_base_seed,
                            "checkpoint_round": p["checkpoint_round"],
                            "transmission_round": p["transmission_round"],
                        })

    output = Path(args.output) if args.output else BENCH / f"experiment_manifest_{args.mode}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} runs to: {output}")
    print(f"Methods={len(p['methods'])}, model_seeds={len(p['model_seeds'])}, targets={len(targets)}, attacks={len(p['attacks'])}, restarts={len(attack_base_seeds)}")


if __name__ == "__main__":
    main()
