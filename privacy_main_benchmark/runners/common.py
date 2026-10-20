from __future__ import annotations
from pathlib import Path
import json

BENCH_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BENCH_DIR.parent


def load_protocol() -> dict:
    return json.loads((BENCH_DIR / "protocol.json").read_text(encoding="utf-8"))


def load_methods() -> dict:
    return json.loads((BENCH_DIR / "configs" / "methods.json").read_text(encoding="utf-8"))


def root_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def benchmark_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else BENCH_DIR / p
