from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attacks.classic_attack import run_classic_attack_experiment
from src.attacks.classic_config import apply_classic_attack_overrides, load_classic_attack_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = parser.parse_args()
    cfg = apply_classic_attack_overrides(load_classic_attack_config(args.config), args.overrides)
    run_classic_attack_experiment(cfg)
