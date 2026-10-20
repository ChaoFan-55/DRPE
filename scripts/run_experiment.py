from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import apply_overrides, build_arg_parser, load_config
from src.runner import run_experiment

if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    run_experiment(cfg)
