from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from runners.common import load_methods, load_protocol, root_path
from runners.command_factory import checkpoint_for, script_for


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Return nonzero when any required path is missing.")
    args = parser.parse_args()

    p = load_protocol()
    methods = load_methods()
    problems = []

    print("[1/3] Attack scripts")
    for method in p["methods"]:
        path = script_for(method)
        ok = path.exists()
        print(f"  {'OK' if ok else 'MISS':4s} {method:10s} {path}")
        if not ok:
            problems.append(f"missing script: {method}: {path}")

    print("[2/3] Round-20 checkpoints")
    for method in p["methods"]:
        for seed in p["model_seeds"]:
            path = checkpoint_for(method, int(seed))
            ok = path.exists()
            print(f"  {'OK' if ok else 'MISS':4s} {method:10s} seed{seed}: {path}")
            if not ok:
                problems.append(f"missing checkpoint: {method} seed{seed}: {path}")

    print("[3/3] DRPE calibration + target set")
    cal = root_path(methods["drpe"]["calibration_path"])
    print(f"  {'OK' if cal.exists() else 'MISS':4s} DRPE calibration: {cal}")
    if not cal.exists():
        problems.append(f"missing DRPE calibration: {cal}")

    target_file = BENCH / "target_samples.csv"
    target_rows = []
    if target_file.exists():
        with target_file.open("r", encoding="utf-8-sig", newline="") as f:
            target_rows = list(csv.DictReader(f))
    target_ok = len(target_rows) == 10 and {int(r['class_id']) for r in target_rows} == set(range(10))
    print(f"  {'OK' if target_ok else 'TODO':4s} target_samples.csv: {len(target_rows)} populated rows")
    if not target_ok:
        print("       Run: python .\\privacy_main_benchmark\\scripts\\build_target_samples.py")

    print("-" * 88)
    if problems:
        print(f"Found {len(problems)} path problem(s). Edit privacy_main_benchmark/configs/methods.json if your local paths differ.")
        if args.strict:
            raise SystemExit(2)
    else:
        print("All required method scripts/checkpoints/calibration paths are present.")


if __name__ == "__main__":
    main()
