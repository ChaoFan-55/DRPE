# -*- coding: utf-8 -*-
"""
plot_figure3_only.py

Figure 3:
更直观易懂版 —— 三列并排水平条形图
(a) Test Accuracy
(b) iDLG LPIPS
(c) Uplink / FedAvg

运行：
python .\scripts\plot_figure3_only.py
"""

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np


# =========================
# 数据
# =========================
METHODS = [
    "FedAvg",
    "Random-k",
    "Top-k",
    "PowerSGD",
    "FLoCoRA",
    "Dense-RP",
    "DRPE",
]

DATA = {
    "FedAvg":   {"acc": 87.41, "lpips": 0.067, "uplink": 1.000},
    "Random-k": {"acc": 87.01, "lpips": 0.026, "uplink": 0.300},
    "Top-k":    {"acc": 86.76, "lpips": 0.054, "uplink": 0.300},
    "PowerSGD": {"acc": 86.55, "lpips": 0.075, "uplink": 0.301},
    "FLoCoRA":  {"acc": 85.94, "lpips": 0.060, "uplink": 0.299},
    "Dense-RP": {"acc": 85.49, "lpips": 0.094, "uplink": 0.301},
    "DRPE":     {"acc": 84.23, "lpips": 0.146, "uplink": 0.301},
}

# 配色：DRPE 主强调，Dense-RP 次强调，其余弱化
COLORS = {
    "FedAvg":   "#7E8A97",
    "Random-k": "#C8D0D8",
    "Top-k":    "#C8D0D8",
    "PowerSGD": "#C8D0D8",
    "FLoCoRA":  "#C8D0D8",
    "Dense-RP": "#7A6B62",
    "DRPE":     "#8D2F3D",
}


# =========================
# 工具函数
# =========================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def pick_cjk_font():
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "PingFang SC",
        "Arial Unicode MS",
    ]
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            return name
    return "DejaVu Sans"


def apply_style():
    cjk_font = pick_cjk_font()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [cjk_font, "DejaVu Sans", "Arial"],
        "axes.unicode_minus": False,

        "figure.dpi": 150,
        "savefig.dpi": 600,

        "axes.linewidth": 1.0,
        "axes.labelsize": 11.5,
        "axes.titlesize": 11.5,

        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        "legend.fontsize": 9,
        "legend.frameon": False,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    return cjk_font


def save_figure_variants(fig, out_base: Path):
    old_svg_fonttype = mpl.rcParams.get("svg.fonttype", "none")

    mpl.rcParams["svg.fonttype"] = "none"
    fig.savefig(str(out_base) + "_editable.svg", bbox_inches="tight")

    mpl.rcParams["svg.fonttype"] = "path"
    fig.savefig(str(out_base) + "_submission.svg", bbox_inches="tight")

    mpl.rcParams["svg.fonttype"] = old_svg_fonttype

    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_base) + ".png", bbox_inches="tight", dpi=600)


# =========================
# 主绘图
# =========================
def plot_figure3(output_dir: Path):
    # 顺序固定：从上到下按论文叙述顺序
    methods = METHODS[::-1]   # barh 中最上方显示 DRPE 更醒目
    y = np.arange(len(methods))

    acc_vals = [DATA[m]["acc"] for m in methods]
    lpips_vals = [DATA[m]["lpips"] for m in methods]
    uplink_vals = [DATA[m]["uplink"] for m in methods]
    colors = [COLORS[m] for m in methods]

    fig, axes = plt.subplots(
        1, 3, figsize=(11.6, 4.8), sharey=True,
        gridspec_kw={"wspace": 0.18}
    )
    ax1, ax2, ax3 = axes

    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # -------------------------
    # (a) Test Accuracy
    # -------------------------
    bars1 = ax1.barh(y, acc_vals, color=colors, edgecolor="white", linewidth=0.8)
    ax1.set_title("(a) Test Accuracy")
    ax1.set_xlabel("Accuracy / %")
    ax1.set_yticks(y)
    ax1.set_yticklabels(methods)
    ax1.set_xlim(83.5, 88.0)

    for i, (bar, v, m) in enumerate(zip(bars1, acc_vals, methods)):
        txt_color = COLORS[m] if m in ("DRPE", "Dense-RP", "FedAvg") else "#58616A"
        fw = "bold" if m in ("DRPE", "Dense-RP", "FedAvg") else "normal"
        ax1.text(
            v + 0.05, bar.get_y() + bar.get_height() / 2,
            "{:.2f}".format(v),
            va="center", ha="left",
            fontsize=9.0, color=txt_color, fontweight=fw
        )

    # -------------------------
    # (b) iDLG LPIPS
    # -------------------------
    bars2 = ax2.barh(y, lpips_vals, color=colors, edgecolor="white", linewidth=0.8)
    ax2.set_title("(b) iDLG LPIPS")
    ax2.set_xlabel("LPIPS ↑")
    ax2.set_xlim(0.0, 0.165)
    ax2.tick_params(axis="y", left=False, labelleft=False)

    for bar, v, m in zip(bars2, lpips_vals, methods):
        txt_color = COLORS[m] if m in ("DRPE", "Dense-RP", "FedAvg") else "#58616A"
        fw = "bold" if m in ("DRPE", "Dense-RP", "FedAvg") else "normal"
        ax2.text(
            v + 0.003, bar.get_y() + bar.get_height() / 2,
            "{:.3f}".format(v),
            va="center", ha="left",
            fontsize=9.0, color=txt_color, fontweight=fw
        )

    # -------------------------
    # (c) Uplink / FedAvg
    # -------------------------
    bars3 = ax3.barh(y, uplink_vals, color=colors, edgecolor="white", linewidth=0.8)
    ax3.set_title("(c) Uplink / FedAvg")
    ax3.set_xlabel("Relative Uplink ↓")
    ax3.set_xlim(0.0, 1.08)
    ax3.tick_params(axis="y", left=False, labelleft=False)

    # 在 c 图加参考线
    ax3.axvline(1.0, color="#9AA3AB", linestyle="--", linewidth=1.0, alpha=0.8)
    ax3.axvline(0.30, color="#C4CBD2", linestyle=":", linewidth=1.0, alpha=0.9)

    for bar, v, m in zip(bars3, uplink_vals, methods):
        txt_color = COLORS[m] if m in ("DRPE", "Dense-RP", "FedAvg") else "#58616A"
        fw = "bold" if m in ("DRPE", "Dense-RP", "FedAvg") else "normal"
        ax3.text(
            v + 0.02, bar.get_y() + bar.get_height() / 2,
            "{:.3f}×".format(v),
            va="center", ha="left",
            fontsize=9.0, color=txt_color, fontweight=fw
        )

    # 额外简短提示
    ax3.text(
        0.30, len(methods) - 0.15, "≈0.30×",
        fontsize=8.8, color="#6A727A", ha="center", va="bottom"
    )
    ax3.text(
        1.00, len(methods) - 0.15, "1.00×",
        fontsize=8.8, color="#6A727A", ha="center", va="bottom"
    )

    # 总体标题尽量简洁
    fig.suptitle(
        "Figure 3  Utility, empirical privacy, and communication under matched uplink budgets",
        y=1.03, fontsize=12.5
    )

    # 底部说明
    fig.text(
        0.5, -0.02,
        "DRPE is highlighted. Dense-RP is shown as the primary structural baseline. "
        "Higher accuracy and LPIPS are better, while lower uplink ratio indicates better communication efficiency.",
        ha="center", va="top", fontsize=9.0, color="#5C646D"
    )

    fig.tight_layout()

    out_base = output_dir / "Figure3_tradeoff"
    save_figure_variants(fig, out_base)
    plt.close(fig)

    return [
        str(out_base) + "_editable.svg",
        str(out_base) + "_submission.svg",
        str(out_base) + ".pdf",
        str(out_base) + ".png",
    ]


def main():
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "figures" / "paper_final"
    ensure_dir(output_dir)

    cjk = apply_style()

    print("Project root:", project_root)
    print("Output dir :", output_dir)
    print("CJK font   :", cjk)

    saved = plot_figure3(output_dir)

    print("\nSaved Figure 3:")
    for p in saved:
        print(" ", p)

    print("\nDone.")


if __name__ == "__main__":
    main()