from __future__ import annotations
import subprocess, sys
from pathlib import Path
BENCH = Path(__file__).resolve().parents[1]
manifest = BENCH / "experiment_manifest_main.csv"
cmd = [sys.executable, str(BENCH / "scripts" / "execute_manifest.py"), "--manifest", str(manifest)] + sys.argv[1:]
raise SystemExit(subprocess.call(cmd, cwd=str(BENCH.parent)))
