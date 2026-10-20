from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]


def _to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def load_metrics(run_dir: Path) -> dict | None:
    classic = run_dir / "classic_attack_metrics.csv"
    if classic.exists():
        with classic.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            return rows[-1]

    candidates = sorted(run_dir.glob("*summary.json"))
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and "psnr" in data and "ssim" in data:
            return data
    return None


def stats(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None, None, 0
    return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0, len(vals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(BENCH / "experiment_manifest_main.csv"))
    args = parser.parse_args()

    with Path(args.manifest).open("r", encoding="utf-8-sig", newline="") as f:
        manifest = list(csv.DictReader(f))

    per_run = []
    missing = []
    for row in manifest:
        run_dir = (
            BENCH / "results" / "raw" / row["method"] / f"seed{int(row['model_seed'])}" /
            row["target_id"] / f"restart{int(row['restart']):02d}" / row["attack"]
        )
        m = load_metrics(run_dir)
        if m is None:
            missing.append(row["run_id"])
            continue
        item = dict(row)
        item.update({
            "psnr": _to_float(m.get("psnr")),
            "ssim": _to_float(m.get("ssim")),
            "lpips": _to_float(m.get("lpips")),
            "label_correct": int(float(m.get("label_correct", 0))) if str(m.get("label_correct", "")).strip() != "" else 0,
            "true_label": m.get("true_label", ""),
            "predicted_label": m.get("predicted_label", ""),
            "best_iteration": m.get("best_iteration", ""),
            "attack_seconds": _to_float(m.get("attack_seconds")),
            "run_dir": str(run_dir),
        })
        per_run.append(item)

    out_dir = BENCH / "results" / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    if per_run:
        fieldnames = list(per_run[0].keys())
        with (out_dir / "per_run_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(per_run)

    groups = defaultdict(list)
    for r in per_run:
        groups[(r["method"], r["attack"])].append(r)

    summary_rows = []
    for (method, attack), rows in sorted(groups.items()):
        pm, ps, pn = stats([r["psnr"] for r in rows])
        sm, ss, sn = stats([r["ssim"] for r in rows])
        lm, ls, ln = stats([r["lpips"] for r in rows])
        label_asr = sum(int(r["label_correct"]) for r in rows) / len(rows) if rows else float("nan")
        summary_rows.append({
            "method": method,
            "attack": attack,
            "n": len(rows),
            "psnr_mean": pm,
            "psnr_std": ps,
            "ssim_mean": sm,
            "ssim_std": ss,
            "lpips_mean": lm,
            "lpips_std": ls,
            "label_asr": label_asr,
        })

    if summary_rows:
        with (out_dir / "privacy_summary_by_method_attack.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader(); w.writerows(summary_rows)

    print(f"Completed runs found: {len(per_run)}/{len(manifest)}")
    print(f"Missing/unparsed runs: {len(missing)}")
    for r in summary_rows:
        def fmt(m, s):
            return "NA" if m is None else f"{m:.3f}±{s:.3f}"
        print(
            f"{r['method']:10s} {r['attack'].upper():4s} n={r['n']:3d} | "
            f"PSNR={fmt(r['psnr_mean'], r['psnr_std'])} | "
            f"SSIM={fmt(r['ssim_mean'], r['ssim_std'])} | "
            f"LPIPS={fmt(r['lpips_mean'], r['lpips_std'])} | "
            f"label_ASR={r['label_asr']:.3f}"
        )
    if missing:
        (out_dir / "missing_runs.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")
    print(f"Saved summaries to: {out_dir}")


if __name__ == "__main__":
    main()
