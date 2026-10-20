from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BENCH = Path(__file__).resolve().parents[1]

METHOD_ORDER = ["fedavg", "random_k", "top_k", "dense_rp", "powersgd", "flocora", "drpe"]
DISPLAY = {
    "fedavg": "FedAvg",
    "random_k": "Random-k",
    "top_k": "Top-k",
    "dense_rp": "Dense-RP",
    "powersgd": "PowerSGD",
    "flocora": "FLoCoRA",
    "drpe": "DRPE",
}
ATTACK_ORDER = ["dlg", "idlg"]


def finite(series: pd.Series) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return x[np.isfinite(x)]


def q(x: np.ndarray, p: float) -> float:
    return float(np.quantile(x, p)) if len(x) else float("nan")


def mean_std(x: np.ndarray) -> tuple[float, float]:
    if len(x) == 0:
        return float("nan"), float("nan")
    return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def bootstrap_mean_ci(values: np.ndarray, seed: int, n_boot: int = 20000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[idx].mean(axis=1)
    return float(values.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        for attack in ATTACK_ORDER:
            g = df[(df.method == method) & (df.attack == attack)]
            if g.empty:
                continue
            row = {
                "method": method,
                "attack": attack,
                "n_runs": len(g),
                "n_targets": g["target_id"].nunique(),
                "n_model_seeds": g["model_seed"].nunique(),
                "n_restarts": g["restart"].nunique(),
            }
            for metric in ["psnr", "ssim", "lpips"]:
                x = finite(g[metric]) if metric in g.columns else np.array([])
                m, s = mean_std(x)
                row.update({
                    f"{metric}_mean": m,
                    f"{metric}_std": s,
                    f"{metric}_median": float(np.median(x)) if len(x) else float("nan"),
                    f"{metric}_q25": q(x, 0.25),
                    f"{metric}_q75": q(x, 0.75),
                    f"{metric}_p90": q(x, 0.90),
                    f"{metric}_p95": q(x, 0.95),
                    f"{metric}_min": float(np.min(x)) if len(x) else float("nan"),
                    f"{metric}_max": float(np.max(x)) if len(x) else float("nan"),
                })
            row["label_asr_run"] = float(pd.to_numeric(g["label_correct"], errors="coerce").mean())
            rows.append(row)
    return pd.DataFrame(rows)


def target_level(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["psnr", "ssim", "lpips", "label_correct"] if c in df.columns]
    agg = (
        df.groupby(["method", "attack", "target_id", "class_id", "class_name"], as_index=False)[cols]
        .mean(numeric_only=True)
    )
    return agg


def paired_drpe(target_df: pd.DataFrame, seed: int, n_boot: int) -> pd.DataFrame:
    rows = []
    for attack in ATTACK_ORDER:
        dr = target_df[(target_df.method == "drpe") & (target_df.attack == attack)].set_index("target_id")
        for comp in METHOD_ORDER:
            if comp == "drpe":
                continue
            co = target_df[(target_df.method == comp) & (target_df.attack == attack)].set_index("target_id")
            common = sorted(set(dr.index) & set(co.index))
            if not common:
                continue
            for metric in ["psnr", "ssim", "lpips"]:
                if metric not in dr.columns or metric not in co.columns:
                    continue
                a = pd.to_numeric(dr.loc[common, metric], errors="coerce").to_numpy(float)
                b = pd.to_numeric(co.loc[common, metric], errors="coerce").to_numpy(float)
                ok = np.isfinite(a) & np.isfinite(b)
                a, b = a[ok], b[ok]
                if len(a) == 0:
                    continue
                diff = a - b  # DRPE - comparator
                m, lo, hi = bootstrap_mean_ci(diff, seed + len(rows) * 17, n_boot=n_boot)
                if metric in {"psnr", "ssim"}:
                    wins = int(np.sum(diff < 0))
                    direction = "negative_favors_drpe"
                else:
                    wins = int(np.sum(diff > 0))
                    direction = "positive_favors_drpe"
                rows.append({
                    "attack": attack,
                    "comparator": comp,
                    "metric": metric,
                    "n_targets": len(diff),
                    "drpe_minus_comparator_mean": m,
                    "bootstrap95_low": lo,
                    "bootstrap95_high": hi,
                    "drpe_target_wins": wins,
                    "direction": direction,
                })
    return pd.DataFrame(rows)


def restart_stability(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        for attack in ATTACK_ORDER:
            g = df[(df.method == method) & (df.attack == attack)]
            for metric in ["psnr", "ssim", "lpips"]:
                if metric not in g.columns or g[metric].notna().sum() == 0:
                    continue
                p = g.pivot_table(index=["model_seed", "target_id"], columns="restart", values=metric, aggfunc="mean")
                if 0 not in p.columns or 1 not in p.columns:
                    continue
                d = (p[1] - p[0]).dropna().to_numpy(float)
                rows.append({
                    "method": method,
                    "attack": attack,
                    "metric": metric,
                    "n_pairs": len(d),
                    "restart1_minus_restart0_mean": float(np.mean(d)),
                    "mean_abs_difference": float(np.mean(np.abs(d))),
                    "max_abs_difference": float(np.max(np.abs(d))),
                })
    return pd.DataFrame(rows)


def _method_labels(methods):
    return [DISPLAY.get(m, m) for m in methods]


def savefig(fig, outbase: Path) -> None:
    outbase.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(outbase.with_suffix(f".{ext}"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def boxplot_target_metric(target_df: pd.DataFrame, metric: str, attack: str, ylabel: str, outbase: Path) -> None:
    methods = [m for m in METHOD_ORDER if not target_df[(target_df.method == m) & (target_df.attack == attack)].empty]
    data = [finite(target_df[(target_df.method == m) & (target_df.attack == attack)][metric]) for m in methods]
    if not any(len(x) for x in data):
        return
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.boxplot(data, labels=_method_labels(methods), showmeans=True, widths=0.58)
    rng = np.random.default_rng(20260822)
    for i, vals in enumerate(data, start=1):
        if len(vals):
            jitter = rng.normal(0, 0.035, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=22, alpha=0.72, zorder=3)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{attack.upper()} target-level distribution (10 targets; seed/restart averaged)")
    ax.grid(axis="y", alpha=0.20)
    ax.tick_params(axis="x", rotation=24)
    fig.tight_layout()
    savefig(fig, outbase)


def ecdf_psnr(df: pd.DataFrame, attack: str, outbase: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    any_data = False
    for method in METHOD_ORDER:
        x = np.sort(finite(df[(df.method == method) & (df.attack == attack)]["psnr"]))
        if not len(x):
            continue
        any_data = True
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", label=DISPLAY.get(method, method))
    if not any_data:
        plt.close(fig); return
    ax.set_xlabel("Reconstruction PSNR (dB)  — higher means stronger leakage")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(f"{attack.upper()} reconstruction PSNR ECDF (60 runs/method)")
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    savefig(fig, outbase)


def label_asr_plot(target_df: pd.DataFrame, attack: str, outbase: Path) -> None:
    methods = [m for m in METHOD_ORDER if not target_df[(target_df.method == m) & (target_df.attack == attack)].empty]
    vals, errs = [], []
    for method in methods:
        x = finite(target_df[(target_df.method == method) & (target_df.attack == attack)]["label_correct"])
        vals.append(float(np.mean(x)) if len(x) else float("nan"))
        errs.append(float(np.std(x, ddof=1)) if len(x) > 1 else 0.0)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    xpos = np.arange(len(methods))
    ax.bar(xpos, vals, yerr=errs, capsize=3)
    ax.set_xticks(xpos)
    ax.set_xticklabels(_method_labels(methods), rotation=24, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Label attack success rate  — lower is better privacy")
    ax.set_title(f"{attack.upper()} label leakage (target-level mean ± SD)")
    ax.grid(axis="y", alpha=0.20)
    fig.tight_layout()
    savefig(fig, outbase)


def tail_plot(desc: pd.DataFrame, attack: str, outbase: Path) -> None:
    g = desc[desc.attack == attack].set_index("method")
    methods = [m for m in METHOD_ORDER if m in g.index]
    p95 = [float(g.loc[m, "psnr_p95"]) for m in methods]
    mx = [float(g.loc[m, "psnr_max"]) for m in methods]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.plot(x, p95, marker="o", label="95th percentile")
    ax.plot(x, mx, marker="s", label="Maximum")
    ax.set_xticks(x)
    ax.set_xticklabels(_method_labels(methods), rotation=24, ha="right")
    ax.set_ylabel("Reconstruction PSNR (dB)  — lower tail risk is better privacy")
    ax.set_title(f"{attack.upper()} high-leakage tail")
    ax.grid(axis="y", alpha=0.20)
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, outbase)


def paired_delta_plot(paired: pd.DataFrame, attack: str, metric: str, outbase: Path) -> None:
    g = paired[(paired.attack == attack) & (paired.metric == metric)].copy()
    if g.empty:
        return
    order = [m for m in METHOD_ORDER if m != "drpe" and m in set(g.comparator)]
    g = g.set_index("comparator").loc[order]
    means = g["drpe_minus_comparator_mean"].to_numpy(float)
    lo = g["bootstrap95_low"].to_numpy(float)
    hi = g["bootstrap95_high"].to_numpy(float)
    yerr = np.vstack([means - lo, hi - means])
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    ax.errorbar(x, means, yerr=yerr, fmt="o", capsize=4)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(_method_labels(order), rotation=24, ha="right")
    if metric in {"psnr", "ssim"}:
        label = f"DRPE − comparator {metric.upper()} (negative favors DRPE)"
    else:
        label = "DRPE − comparator LPIPS (positive favors DRPE)"
    ax.set_ylabel(label)
    ax.set_title(f"{attack.upper()} paired target-level effect (95% target bootstrap CI)")
    ax.grid(axis="y", alpha=0.20)
    fig.tight_layout()
    savefig(fig, outbase)


def main() -> None:
    parser = argparse.ArgumentParser(description="Target-aware privacy statistics and publication figures.")
    parser.add_argument("--per-run", default=str(BENCH / "results" / "summaries" / "per_run_metrics.csv"))
    parser.add_argument("--outdir", default=str(BENCH / "results" / "statistics_v2"))
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    df = pd.read_csv(args.per_run)
    expected = {"method", "attack", "target_id", "class_id", "class_name", "model_seed", "restart", "psnr", "ssim", "label_correct"}
    miss = expected.difference(df.columns)
    if miss:
        raise SystemExit(f"Missing required columns: {sorted(miss)}")
    if "lpips" not in df.columns:
        df["lpips"] = np.nan

    outdir = Path(args.outdir)
    table_dir = outdir / "tables"
    fig_dir = outdir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    desc = descriptive_table(df)
    target = target_level(df)
    paired = paired_drpe(target, seed=args.seed, n_boot=args.bootstrap)
    restart = restart_stability(df)

    desc.to_csv(table_dir / "method_attack_descriptive.csv", index=False, encoding="utf-8-sig")
    target.to_csv(table_dir / "target_level_metrics.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(table_dir / "drpe_paired_target_comparisons.csv", index=False, encoding="utf-8-sig")
    restart.to_csv(table_dir / "restart_stability.csv", index=False, encoding="utf-8-sig")

    # Compact paper-ready table.
    paper_cols = [
        "method", "attack", "n_runs", "n_targets",
        "psnr_mean", "psnr_std", "psnr_median", "psnr_p95", "psnr_max",
        "ssim_mean", "ssim_std", "ssim_median", "ssim_p95", "ssim_max",
        "lpips_mean", "lpips_std", "lpips_median", "lpips_q25", "lpips_q75",
        "label_asr_run",
    ]
    desc[[c for c in paper_cols if c in desc.columns]].to_csv(
        table_dir / "privacy_main_table_v2.csv", index=False, encoding="utf-8-sig"
    )

    for attack in ATTACK_ORDER:
        boxplot_target_metric(target, "psnr", attack, "Reconstruction PSNR (dB) — lower is better privacy", fig_dir / f"psnr_target_box_{attack}")
        boxplot_target_metric(target, "ssim", attack, "Reconstruction SSIM — lower is better privacy", fig_dir / f"ssim_target_box_{attack}")
        if target["lpips"].notna().any():
            boxplot_target_metric(target, "lpips", attack, "LPIPS — higher is better privacy", fig_dir / f"lpips_target_box_{attack}")
        ecdf_psnr(df, attack, fig_dir / f"psnr_ecdf_{attack}")
        label_asr_plot(target, attack, fig_dir / f"label_asr_{attack}")
        tail_plot(desc, attack, fig_dir / f"psnr_tail_{attack}")
        paired_delta_plot(paired, attack, "psnr", fig_dir / f"drpe_paired_delta_psnr_{attack}")
        paired_delta_plot(paired, attack, "ssim", fig_dir / f"drpe_paired_delta_ssim_{attack}")
        if target["lpips"].notna().any():
            paired_delta_plot(paired, attack, "lpips", fig_dir / f"drpe_paired_delta_lpips_{attack}")

    lpips_count = int(pd.to_numeric(df["lpips"], errors="coerce").notna().sum())
    print(f"Input runs: {len(df)}")
    print(f"LPIPS available: {lpips_count}/{len(df)}")
    print(f"Target-level rows: {len(target)} (expected 7 methods × 2 attacks × 10 targets = 140)")
    print(f"Saved tables to: {table_dir}")
    print(f"Saved figures to: {fig_dir}")
    if lpips_count == 0:
        print("NOTE: LPIPS figures were skipped. Run compute_lpips_offline.py --update-per-run first, then rerun this script.")


if __name__ == "__main__":
    main()
