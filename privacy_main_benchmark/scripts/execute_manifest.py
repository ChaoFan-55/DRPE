from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from runners.command_factory import build_command


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def has_result(output_dir: Path) -> bool:
    if not output_dir.exists():
        return False
    if (output_dir / "run_meta.json").exists():
        try:
            meta = json.loads((output_dir / "run_meta.json").read_text(encoding="utf-8"))
            if meta.get("status") == "success":
                return True
        except Exception:
            pass
    # Backward-compatible detection in case a run finished before meta was written.
    patterns = ["classic_attack_metrics.csv", "*attack*summary.json", "*_summary.json"]
    return any(any(output_dir.glob(pattern)) for pattern in patterns)


def format_cmd(cmd: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    return shlex.join(cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="Rerun even when a success marker/result already exists.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--model-seed", type=int, action="append", default=[])
    parser.add_argument("--attack", action="append", choices=["dlg", "idlg"], default=[])
    parser.add_argument("--max-runs", type=int, default=0)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    rows = read_manifest(manifest)
    if args.method:
        rows = [r for r in rows if r["method"] in set(args.method)]
    if args.model_seed:
        rows = [r for r in rows if int(r["model_seed"]) in set(args.model_seed)]
    if args.attack:
        rows = [r for r in rows if r["attack"] in set(args.attack)]
    if args.max_runs > 0:
        rows = rows[: args.max_runs]

    print(f"Manifest: {manifest}")
    print(f"Selected runs: {len(rows)}")
    if not rows:
        return

    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    success = skipped = failed = 0
    for idx, row in enumerate(rows, 1):
        cmd, output_dir = build_command(row)
        if not args.no_resume and has_result(output_dir):
            skipped += 1
            print(f"[{idx}/{len(rows)}] SKIP {row['run_id']}")
            continue

        print("=" * 110)
        print(f"[{idx}/{len(rows)}] {row['run_id']}")
        print(format_cmd(cmd))
        if args.dry_run:
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_command.txt").write_text(format_cmd(cmd) + "\n", encoding="utf-8")
        meta = {
            "run_id": row["run_id"],
            "status": "running",
            "row": row,
            "command": cmd,
            "started_at_unix": time.time(),
        }
        (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        log_path = output_dir / "console.log"
        start = time.perf_counter()
        try:
            with log_path.open("w", encoding="utf-8", errors="replace") as log:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(BENCH.parent),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line, end="")
                    log.write(line)
                rc = proc.wait()
            elapsed = time.perf_counter() - start
            meta.update({"finished_at_unix": time.time(), "elapsed_seconds": elapsed, "return_code": rc})
            if rc == 0:
                meta["status"] = "success"
                success += 1
            else:
                meta["status"] = "failed"
                failed += 1
            (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            if rc != 0 and not args.continue_on_error:
                raise SystemExit(rc)
        except KeyboardInterrupt:
            meta.update({"status": "interrupted", "finished_at_unix": time.time()})
            (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            raise

    print("=" * 110)
    if args.dry_run:
        print(f"Dry-run complete: {len(rows)} command(s) generated.")
    else:
        print(f"Done. success={success}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
