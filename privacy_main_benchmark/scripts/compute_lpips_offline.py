from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

BENCH = Path(__file__).resolve().parents[1]


def _load_rgb01(path: Path):
    import torch
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _png_psnr(a, b) -> float:
    import torch
    mse = torch.mean((a - b) ** 2).item()
    if mse <= 0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def _run_dir(row: pd.Series) -> Path:
    return (
        BENCH / "results" / "raw" / str(row["method"]) /
        f"seed{int(row['model_seed'])}" / str(row["target_id"]) /
        f"restart{int(row['restart']):02d}" / str(row["attack"])
    )


def _unique_best(candidates: list[Path], kind: str, run_id: str) -> Path:
    candidates = sorted({p.resolve() for p in candidates if p.is_file()})
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"{run_id}: no {kind} image found")
    # Prefer exact reconstruction files over comparison/figure files.
    if kind == "reconstruction":
        exact = [p for p in candidates if p.name.startswith("reconstruction_") and p.name.endswith("_best.png")]
        if len(exact) == 1:
            return exact[0]
    raise RuntimeError(f"{run_id}: ambiguous {kind} images: {[str(p) for p in candidates]}")


def find_pair(row: pd.Series) -> tuple[Path, Path]:
    run_id = str(row["run_id"])
    run_dir = _run_dir(row)
    if not run_dir.exists():
        raise FileNotFoundError(f"{run_id}: run directory not found: {run_dir}")

    target_candidates = []
    direct_target = run_dir / "target.png"
    if direct_target.exists():
        target_candidates.append(direct_target)
    sample_target = run_dir / f"sample_{int(row['sample_index']):05d}" / "target.png"
    if sample_target.exists():
        target_candidates.append(sample_target)
    if not target_candidates:
        target_candidates.extend(run_dir.rglob("target.png"))

    recon_candidates = list(run_dir.glob("reconstruction_*_best.png"))
    if not recon_candidates:
        recon_candidates = list(run_dir.rglob("reconstruction_*_best.png"))

    target = _unique_best(target_candidates, "target", run_id)
    recon = _unique_best(recon_candidates, "reconstruction", run_id)
    return target, recon


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute standard LPIPS offline from saved target/best PNGs.")
    parser.add_argument(
        "--per-run",
        default=str(BENCH / "results" / "summaries" / "per_run_metrics.csv"),
    )
    parser.add_argument(
        "--output",
        default=str(BENCH / "results" / "summaries" / "lpips_offline.csv"),
    )
    parser.add_argument("--net", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--device", default="auto", help="auto/cuda/cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--update-per-run", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--psnr-check-tol", type=float, default=0.50,
                        help="Warn if PSNR from saved PNG differs from recorded PSNR by more than this many dB.")
    args = parser.parse_args()

    try:
        import torch
        import lpips
    except Exception as exc:
        raise SystemExit(
            "LPIPS dependency is unavailable. Install it first with:\n"
            "  python -m pip install lpips\n"
            f"Original import error: {exc}"
        )

    per_run_path = Path(args.per_run)
    df = pd.read_csv(per_run_path)
    required = {"run_id", "method", "model_seed", "target_id", "sample_index", "restart", "attack", "psnr"}
    missing_cols = required.difference(df.columns)
    if missing_cols:
        raise SystemExit(f"Missing required columns in {per_run_path}: {sorted(missing_cols)}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Rows: {len(df)} | LPIPS net={args.net} | device={device} | batch_size={args.batch_size}")
    print("Metric source: saved target.png and reconstruction_*_best.png; tensors are mapped from [0,1] to [-1,1].")

    model = lpips.LPIPS(net=args.net).to(device).eval()

    records: list[dict] = []
    pending: list[tuple[int, pd.Series, Path, Path, object, object, float]] = []
    missing_runs: list[str] = []

    def flush_batch() -> None:
        nonlocal pending
        if not pending:
            return
        ta = torch.stack([x[4] for x in pending]).to(device)
        rb = torch.stack([x[5] for x in pending]).to(device)
        # Official LPIPS expects RGB tensors in [-1,1].
        with torch.no_grad():
            vals = model(ta * 2.0 - 1.0, rb * 2.0 - 1.0)
        vals = vals.reshape(len(pending), -1).mean(dim=1).detach().cpu().numpy()
        for item, val in zip(pending, vals):
            idx, row, target, recon, _t, _r, png_psnr = item
            recorded_psnr = float(row["psnr"]) if pd.notna(row["psnr"]) else float("nan")
            delta = png_psnr - recorded_psnr if math.isfinite(recorded_psnr) else float("nan")
            records.append({
                "run_id": row["run_id"],
                "method": row["method"],
                "model_seed": int(row["model_seed"]),
                "target_id": row["target_id"],
                "attack": row["attack"],
                "restart": int(row["restart"]),
                "lpips": float(val),
                "target_png": str(target),
                "reconstruction_png": str(recon),
                "recorded_psnr": recorded_psnr,
                "png_psnr": png_psnr,
                "png_minus_recorded_psnr": delta,
            })
        pending = []

    for idx, row in df.iterrows():
        run_id = str(row["run_id"])
        try:
            target, recon = find_pair(row)
            t = _load_rgb01(target)
            r = _load_rgb01(recon)
            if t.shape != r.shape:
                raise RuntimeError(f"image shape mismatch: target={tuple(t.shape)}, recon={tuple(r.shape)}")
            png_psnr = _png_psnr(t, r)
            pending.append((idx, row, target, recon, t, r, png_psnr))
            if len(pending) >= args.batch_size:
                flush_batch()
        except Exception as exc:
            missing_runs.append(run_id)
            print(f"[MISSING] {run_id}: {exc}")
            if not args.allow_missing:
                # Continue scanning so user sees all missing pairs, but refuse final success below.
                continue
    flush_batch()

    out = pd.DataFrame(records)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    if len(out):
        abs_delta = out["png_minus_recorded_psnr"].abs().dropna()
        print(f"Image pairs found: {len(out)}/{len(df)}")
        if len(abs_delta):
            print(
                "Saved-PNG PSNR sanity check | "
                f"mean |Δ|={abs_delta.mean():.4f} dB, "
                f"p95 |Δ|={abs_delta.quantile(0.95):.4f} dB, "
                f"max |Δ|={abs_delta.max():.4f} dB"
            )
            bad = out[abs(out["png_minus_recorded_psnr"]) > args.psnr_check_tol]
            if len(bad):
                print(f"WARNING: {len(bad)} run(s) exceed PSNR sanity tolerance {args.psnr_check_tol:.2f} dB.")
                bad.to_csv(out_path.with_name("lpips_png_psnr_sanity_warnings.csv"), index=False, encoding="utf-8-sig")

    if missing_runs:
        miss_path = out_path.with_name("lpips_missing_runs.txt")
        miss_path.write_text("\n".join(missing_runs) + "\n", encoding="utf-8")
        print(f"Missing image pairs: {len(missing_runs)} -> {miss_path}")
        if not args.allow_missing:
            raise SystemExit("LPIPS computation is incomplete. Fix missing/ambiguous image paths before updating per_run_metrics.csv.")

    if args.update_per_run:
        if len(out) != len(df) or out["run_id"].nunique() != len(df):
            raise SystemExit("Refusing to update per_run_metrics.csv because LPIPS coverage is not exactly complete.")
        mapping = out.set_index("run_id")["lpips"]
        if not set(df["run_id"]).issubset(set(mapping.index)):
            raise SystemExit("LPIPS mapping does not cover all per-run rows.")
        backup = per_run_path.with_name(per_run_path.stem + ".pre_lpips.csv")
        if not backup.exists():
            shutil.copy2(per_run_path, backup)
            print(f"Backup created: {backup}")
        df["lpips"] = df["run_id"].map(mapping).astype(float)
        df.to_csv(per_run_path, index=False, encoding="utf-8-sig")
        print(f"Updated: {per_run_path}")

    print(f"Saved LPIPS details to: {out_path}")


if __name__ == "__main__":
    main()
