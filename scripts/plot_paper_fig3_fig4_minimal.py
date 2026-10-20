# -*- coding: utf-8 -*-
"""
DRPE paper figures - minimalist final version

Figure 3:
    Accuracy vs iDLG LPIPS under matched uplink budget.
    - No bubble encoding.
    - FedAvg shown as hollow marker with 1.00x uplink.
    - All compressed methods are ~0.30x uplink.
    - DRPE is the only primary visual emphasis.

Figure 4:
    (a) iDLG Label ASR
    (b) iDLG LPIPS (mean ± std)
    (c) iDLG PSNR ECDF from real per_run_metrics.csv

Outputs:
    *_editable.svg
    *_submission.svg
    *.pdf
    *.png (600 dpi)

Recommended location:
    drpe_formal_experiments/scripts/plot_paper_fig3_fig4.py
"""

from __future__ import print_function

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

METHODS = [
    "FedAvg",
    "Random-k",
    "Top-k",
    "PowerSGD",
    "FLoCoRA",
    "Dense-RP",
    "DRPE",
]

TRADEOFF = {
    "FedAvg":   {"acc": 87.41, "lpips": 0.067, "uplink": 1.000},
    "Random-k": {"acc": 87.01, "lpips": 0.026, "uplink": 0.300},
    "Top-k":    {"acc": 86.76, "lpips": 0.054, "uplink": 0.300},
    "PowerSGD": {"acc": 86.55, "lpips": 0.075, "uplink": 0.301},
    "FLoCoRA":  {"acc": 85.94, "lpips": 0.060, "uplink": 0.299},
    "Dense-RP": {"acc": 85.49, "lpips": 0.094, "uplink": 0.301},
    "DRPE":     {"acc": 84.23, "lpips": 0.146, "uplink": 0.301},
}

IDLG_ASR = {
    "FedAvg": 100.0,
    "Random-k": 100.0,
    "Top-k": 100.0,
    "PowerSGD": 100.0,
    "FLoCoRA": 100.0,
    "Dense-RP": 60.0,
    "DRPE": 56.7,
}

IDLG_LPIPS = {
    "FedAvg":   (0.067, 0.078),
    "Random-k": (0.026, 0.054),
    "Top-k":    (0.054, 0.064),
    "PowerSGD": (0.075, 0.081),
    "FLoCoRA":  (0.060, 0.056),
    "Dense-RP": (0.094, 0.111),
    "DRPE":     (0.146, 0.147),
}

# Minimal palette: baseline gray-blue, Dense-RP dark gray, DRPE burgundy.
BASELINE = "#A8B1BA"
FEDAVG_EDGE = "#727C86"
DENSE = "#5E6670"
DRPE = "#9A3342"
GRID = "#D9DEE3"
TEXT = "#2E343B"

MARKERS = {
    "FedAvg": "o",
    "Random-k": "s",
    "Top-k": "D",
    "PowerSGD": "^",
    "FLoCoRA": "P",
    "Dense-RP": "v",
    "DRPE": "*",
}


# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------

def pick_cjk_font():
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "PingFang SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            return name
    return "DejaVu Sans"


def apply_style():
    font = pick_cjk_font()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font, "DejaVu Sans", "Arial"],
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.9,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
    return font


def clean_axis(ax, grid_axis="both"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#77808A")
    ax.spines["bottom"].set_color("#77808A")
    ax.tick_params(colors=TEXT, width=0.8)
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.55,
            color=GRID, alpha=0.7)
    ax.set_axisbelow(True)


def save_variants(fig, base):
    old = mpl.rcParams.get("svg.fonttype", "none")

    mpl.rcParams["svg.fonttype"] = "none"
    fig.savefig(str(base) + "_editable.svg", bbox_inches="tight")

    mpl.rcParams["svg.fonttype"] = "path"
    fig.savefig(str(base) + "_submission.svg", bbox_inches="tight")

    mpl.rcParams["svg.fonttype"] = old
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".png", bbox_inches="tight", dpi=600)


# -----------------------------------------------------------------------------
# Real PSNR ECDF loader
# -----------------------------------------------------------------------------

def norm_name(x):
    return str(x).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def canonical_method(x):
    m = norm_name(x)
    mapping = {
        "fedavg": "FedAvg",
        "randomk": "Random-k",
        "topk": "Top-k",
        "powersgd": "PowerSGD",
        "flocora": "FLoCoRA",
        "denserp": "Dense-RP",
        "drpe": "DRPE",
    }
    return mapping.get(m, str(x))


def find_col(df, candidates):
    lookup = {norm_name(c): c for c in df.columns}
    for c in candidates:
        if norm_name(c) in lookup:
            return lookup[norm_name(c)]
    raise KeyError("Cannot find any of columns: {}".format(candidates))


def find_per_run_csv(root):
    candidates = [
        root / "privacy_main_benchmark" / "results" / "summaries" / "per_run_metrics.csv",
        root / "privacy_main_benchmark" / "results" / "per_run_metrics.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_idlg_psnr(root):
    path = find_per_run_csv(root)
    if path is None:
        return None, None

    df = pd.read_csv(path)
    method_col = find_col(df, ["method", "algorithm", "scheme", "defense_method"])
    attack_col = find_col(df, ["attack", "attack_type", "attack_name"])
    psnr_col = find_col(df, ["psnr", "best_psnr", "final_psnr", "metric_psnr"])

    tmp = df.copy()
    tmp["__method"] = tmp[method_col].map(canonical_method)
    tmp["__attack"] = tmp[attack_col].astype(str).str.lower()
    tmp = tmp[tmp["__attack"].str.contains("idlg", na=False)]

    result = {}
    for method in METHODS:
        vals = tmp.loc[tmp["__method"] == method, psnr_col].dropna().astype(float).values
        if len(vals) > 0:
            result[method] = np.sort(vals)

    return path, result


# -----------------------------------------------------------------------------
# Figure 3
# -----------------------------------------------------------------------------

def plot_figure3(output_dir):
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    clean_axis(ax)

    ax.set_xlabel("测试准确率 / % ↑", color=TEXT)
    ax.set_ylabel("iDLG LPIPS ↑", color=TEXT)
    ax.set_xlim(83.7, 87.8)
    ax.set_ylim(0.015, 0.158)

    # Soft guide for compressed methods. No large annotation box.
    ax.text(
        0.02, 0.97,
        "压缩方法：≈0.30× FedAvg 上行通信",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=8.8, color="#66717B"
    )

    # Baselines
    for method in METHODS:
        d = TRADEOFF[method]
        x, y = d["acc"], d["lpips"]

        if method == "FedAvg":
            ax.scatter(x, y, s=78, marker="o", facecolors="white",
                       edgecolors=FEDAVG_EDGE, linewidths=1.4, zorder=3)
        elif method == "DRPE":
            ax.scatter(x, y, s=165, marker="*", c=DRPE,
                       edgecolors="white", linewidths=0.9, zorder=6)
        elif method == "Dense-RP":
            ax.scatter(x, y, s=82, marker=MARKERS[method], c=DENSE,
                       edgecolors="white", linewidths=0.8, zorder=5)
        else:
            ax.scatter(x, y, s=58, marker=MARKERS[method], c=BASELINE,
                       edgecolors="white", linewidths=0.8, zorder=4)

    # Only necessary labels; other methods remain legible via compact legend.
    ax.annotate("DRPE", xy=(84.23, 0.146), xytext=(7, 4),
                textcoords="offset points", fontsize=10.2,
                fontweight="bold", color=DRPE)
    ax.annotate("Dense-RP", xy=(85.49, 0.094), xytext=(7, 3),
                textcoords="offset points", fontsize=8.9,
                color=DENSE)
    ax.annotate("FedAvg  1.00×", xy=(87.41, 0.067), xytext=(-2, 9),
                textcoords="offset points", fontsize=8.8,
                color=FEDAVG_EDGE, ha="right")
    ax.annotate("Random-k", xy=(87.01, 0.026), xytext=(-5, -14),
                textcoords="offset points", fontsize=8.6,
                color="#6D7781", ha="right")

    # Minimal legend for unlabeled baselines only.
    legend_handles = [
        Line2D([0], [0], marker="D", linestyle="none", markersize=6,
               markerfacecolor=BASELINE, markeredgecolor="white", label="Top-k"),
        Line2D([0], [0], marker="^", linestyle="none", markersize=6,
               markerfacecolor=BASELINE, markeredgecolor="white", label="PowerSGD"),
        Line2D([0], [0], marker="P", linestyle="none", markersize=6,
               markerfacecolor=BASELINE, markeredgecolor="white", label="FLoCoRA"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", frameon=False,
              handletextpad=0.4, borderpad=0.1, labelspacing=0.35)

    fig.tight_layout()
    base = output_dir / "Figure3_tradeoff"
    save_variants(fig, base)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 4
# -----------------------------------------------------------------------------

def plot_figure4(project_root, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.75))
    ax1, ax2, ax3 = axes

    for ax in axes:
        clean_axis(ax, grid_axis="y")

    x = np.arange(len(METHODS))
    labels = METHODS

    # -------------------- (a) Label ASR --------------------
    vals = [IDLG_ASR[m] for m in METHODS]
    colors = []
    for m in METHODS:
        if m == "DRPE":
            colors.append(DRPE)
        elif m == "Dense-RP":
            colors.append(DENSE)
        else:
            colors.append(BASELINE)

    bars = ax1.bar(x, vals, width=0.68, color=colors,
                   edgecolor="white", linewidth=0.7)

    ax1.set_ylabel("Label ASR / % ↓", color=TEXT)
    ax1.set_ylim(0, 108)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax1.text(0.02, 0.97, "(a)", transform=ax1.transAxes,
             ha="left", va="top", fontsize=10.5, fontweight="bold")

    # Only annotate the informative contrast.
    for method in ["Dense-RP", "DRPE"]:
        i = METHODS.index(method)
        ax1.text(i, vals[i] + 2.0, "{:.1f}".format(vals[i]),
                 ha="center", va="bottom", fontsize=9,
                 color=DENSE if method == "Dense-RP" else DRPE,
                 fontweight="bold")

    # -------------------- (b) LPIPS --------------------
    means = [IDLG_LPIPS[m][0] for m in METHODS]
    stds = [IDLG_LPIPS[m][1] for m in METHODS]

    # Baselines first
    for i, m in enumerate(METHODS):
        if m == "DRPE":
            c, ms, lw, z = DRPE, 9.5, 1.8, 5
        elif m == "Dense-RP":
            c, ms, lw, z = DENSE, 7.0, 1.5, 4
        else:
            c, ms, lw, z = BASELINE, 5.8, 1.0, 3

        ax2.errorbar(
            i, means[i], yerr=stds[i], fmt=MARKERS[m],
            color=c, ecolor=c, markersize=ms,
            markerfacecolor=c, markeredgecolor="white",
            markeredgewidth=0.7, elinewidth=lw,
            capsize=3.0, capthick=lw, zorder=z
        )

    ax2.set_ylabel("LPIPS ↑", color=TEXT)
    ax2.set_ylim(0, 0.31)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right")
    ax2.text(0.02, 0.97, "(b)", transform=ax2.transAxes,
             ha="left", va="top", fontsize=10.5, fontweight="bold")

    # Only the two main structural comparisons receive numeric labels.
    for method in ["Dense-RP", "DRPE"]:
        i = METHODS.index(method)
        ax2.text(i, means[i] + stds[i] + 0.010,
                 "{:.3f}".format(means[i]),
                 ha="center", va="bottom", fontsize=8.8,
                 color=DENSE if method == "Dense-RP" else DRPE,
                 fontweight="bold")

    # -------------------- (c) Real PSNR ECDF --------------------
    path, ecdf = load_idlg_psnr(project_root)

    if path is not None and ecdf:
        print("Figure 4(c) ECDF source: {}".format(path))

        for method in METHODS:
            if method not in ecdf:
                continue
            vals_m = np.asarray(ecdf[method], dtype=float)
            y_ecdf = np.arange(1, len(vals_m) + 1, dtype=float) / float(len(vals_m))

            if method == "DRPE":
                color, lw, alpha, ls, z = DRPE, 2.5, 1.0, "-", 6
            elif method == "Dense-RP":
                color, lw, alpha, ls, z = DENSE, 1.9, 0.95, "--", 5
            elif method == "FedAvg":
                color, lw, alpha, ls, z = "#7C8791", 1.5, 0.90, "-", 4
            else:
                color, lw, alpha, ls, z = "#B8C0C7", 1.0, 0.65, "-", 2

            ax3.step(vals_m, y_ecdf, where="post", color=color,
                     linewidth=lw, alpha=alpha, linestyle=ls,
                     label=method, zorder=z)
    else:
        ax3.text(0.5, 0.5, "未找到 per_run_metrics.csv\n无法绘制真实 ECDF",
                 transform=ax3.transAxes, ha="center", va="center",
                 fontsize=9.2, color=DRPE)

    ax3.set_xlabel("PSNR / dB ↓", color=TEXT)
    ax3.set_ylabel("ECDF", color=TEXT)
    ax3.set_xlim(4, 28)
    ax3.set_ylim(0, 1.02)
    ax3.text(0.02, 0.97, "(c)", transform=ax3.transAxes,
             ha="left", va="top", fontsize=10.5, fontweight="bold")

    # Only show key curves in legend; gray curves represent remaining baselines.
    handles = [
        Line2D([0], [0], color=DRPE, lw=2.5, label="DRPE"),
        Line2D([0], [0], color=DENSE, lw=1.9, ls="--", label="Dense-RP"),
        Line2D([0], [0], color="#7C8791", lw=1.5, label="FedAvg"),
        Line2D([0], [0], color="#B8C0C7", lw=1.0, label="其他压缩基线"),
    ]
    ax3.legend(handles=handles, loc="lower right", frameon=False,
               handlelength=2.4, handletextpad=0.5, labelspacing=0.35)

    # No overall title inside the image; use manuscript caption instead.
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.27, top=0.96, wspace=0.33)

    base = output_dir / "Figure4_privacy"
    save_variants(fig, base)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", choices=["3", "4", "all"], default="all")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "figures" / "paper_final"
    output_dir.mkdir(parents=True, exist_ok=True)

    font = apply_style()
    print("Project root: {}".format(project_root))
    print("Output dir : {}".format(output_dir))
    print("CJK font   : {}".format(font))

    if args.figure in ("3", "all"):
        plot_figure3(output_dir)
        print("Saved Figure 3.")

    if args.figure in ("4", "all"):
        plot_figure4(project_root, output_dir)
        print("Saved Figure 4.")

    print("Done.")
    print("Recommended manuscript files:")
    print("  Figure3_tradeoff_submission.svg")
    print("  Figure4_privacy_submission.svg")
    print("Editable versions are also saved as *_editable.svg")


if __name__ == "__main__":
    main()
