"""从现有结果绘制论文图3—图6；只读取数据，不导入训练代码。

示例（在实验项目根目录执行）：
python plot_figures_3_to_6.py --root . --output-dir figures/revision_fig3_6

依赖：numpy、matplotlib。默认同时输出SVG、PDF、PNG。
所有标准差均为样本标准差(ddof=1)，不是标准误或置信区间。
图3/4：三个训练种子；图5/6：60次攻击运行的描述统计。
排版：子图标题居中放在横轴下方，图例位于子图标题下方，统计说明置于整图底部。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

SEEDS = (42, 43, 44)
METHODS = ("fedavg", "random_k", "top_k", "powersgd", "flocora", "dense_rp", "drpe")
NAMES = {
    "fedavg": "FedAvg", "random_k": "Random-k", "top_k": "Top-k",
    "powersgd": "PowerSGD", "flocora": "FLoCoRA", "dense_rp": "Dense-RP",
    "drpe": "DRPE", "random_k_clip_noise": "Random-k＋裁剪＋噪声",
    "static": "固定随机状态＋裁剪＋噪声", "clip_only": "动态投影＋裁剪，无噪声",
}
COLORS = {
    "fedavg": "#656565", "random_k": "#2878B5", "top_k": "#2B927D",
    "powersgd": "#8B6BA8", "flocora": "#C19835", "dense_rp": "#698495",
    "drpe": "#C64F3D", "random_k_clip_noise": "#D38B38",
    "static": "#817093", "clip_only": "#2878B5",
}
MARKERS = ("o", "s", "^", "D", "v", "P", "*", "X")
SOURCES = {}
VALUES = {}


def setup_style(font_path=None):
    candidates = ([Path(font_path)] if font_path else []) + [
        Path("C:/Windows/Fonts/simsun.ttc"), Path("C:/Windows/Fonts/msyh.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    found = next((p for p in candidates if p.is_file()), None)
    if found is None:
        raise FileNotFoundError("没有找到中文字体。请用 --font 指定中文 TTF/OTF/TTC 字体路径。")
    font_manager.fontManager.addfont(str(found))
    family = font_manager.FontProperties(fname=str(found)).get_name()
    plt.rcParams.update({
        "font.family": family, "font.size": 10.5, "axes.titlesize": 11.5,
        "axes.labelsize": 10.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "axes.unicode_minus": False,
        "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "path", "axes.linewidth": 0.8,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def track(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"缺少数据文件：{path}")
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(path):
    return json.loads(track(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with track(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def numbers(rows, field):
    try:
        result = np.array([float(row[field]) for row in rows], dtype=float)
    except (KeyError, ValueError) as e:
        raise ValueError(f"指标缺失或不是数字：{field}") from e
    if not np.isfinite(result).all():
        raise ValueError(f"指标包含 NaN/Inf：{field}")
    return result


def stat(a):
    a = np.asarray(a, dtype=float)
    return {"n": int(len(a)), "mean": float(a.mean()), "sample_sd": float(a.std(ddof=1))}


def train_dir(root, method, seed):
    if method == "static":
        return root / "outputs/ablation_dynamic_static" / f"static_seed{seed}"
    if method == "clip_only":
        return root / "outputs/ablation_components/projection_clip_only" / f"seed{seed}"
    suffix = f"seed{seed}" if method in ("fedavg", "drpe") else f"commmatched_seed{seed}"
    return root / "outputs/cifar10/resnet18" / method / suffix


def training(root, method):
    curves, accuracies, uplinks = [], [], []
    for seed in SEEDS:
        folder = train_dir(root, method, seed)
        summary = read_json(folder / "summary.json")
        rows = read_csv(folder / "metrics.csv")
        rounds = numbers(rows, "round")
        if not np.array_equal(rounds, np.arange(1, 201)):
            raise ValueError(f"{folder}: 训练轮次不完整，预期1—200轮。")
        vals = numbers(rows, "val_accuracy")
        accuracy = float(summary["test_accuracy"])
        if not np.isfinite(accuracy) or not 0 <= accuracy <= 100:
            raise ValueError(f"{folder}: 测试准确率异常。")
        if int(summary["seed"]) != seed:
            raise ValueError(f"{folder}: 种子不一致。")
        if abs(vals.max() - float(summary["best_validation_accuracy"])) > 1e-6:
            raise ValueError(f"{folder}: 最优验证准确率与汇总不一致。")
        uplink = numbers(rows, "round_uplink_mib")
        if abs(uplink.sum() - float(summary["cumulative_uplink_mib"])) > 1e-4:
            raise ValueError(f"{folder}: 累计通信量与逐轮记录不一致。")
        curves.append(vals)
        accuracies.append(accuracy)
        uplinks.append(uplink.mean())
    result = (np.array(curves), np.array(accuracies), np.array(uplinks))
    VALUES.setdefault("training", {})[method] = {
        "seeds": list(SEEDS), "test_accuracy_by_seed": accuracies,
        "test_accuracy": stat(accuracies), "uplink_mib_per_round": stat(uplinks),
    }
    return result


def clean_axis(ax, axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color="#E4E6E8", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def panel(ax, title):
    # 等待坐标标签完成排版后，再将子图标题放到横轴下方。
    ax._paper_panel_caption = title
    clean_axis(ax)


def place_bottom_captions(fig):
    """按实际文字边界布置子图标题和图例，避免压住横轴标签。

    同一行子图的标题对齐；尺寸按英寸指定，导出分辨率不会改变间距。
    """
    height = fig.get_figheight()
    legends = {}
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            # 优先保留原始句柄，使误差棒图例中的点形和误差线完整保留。
            handles, labels = ax.get_legend_handles_labels()
            if not handles:  # 手工创建的训练种子图例
                handles = getattr(legend, "legend_handles", None)
                if handles is None:  # 兼容较早的Matplotlib版本
                    handles = legend.legendHandles
                labels = [text.get_text() for text in legend.get_texts()]
            legends[ax] = (list(handles), labels)
            legend.remove()

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    rows = {}
    for ax in fig.axes:
        if hasattr(ax, "_paper_panel_caption"):
            rows.setdefault(round(ax.get_position().y0, 5), []).append(ax)

    for row_axes in rows.values():
        label_bottom = min(
            ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).y0
            for ax in row_axes
        )
        caption_top = label_bottom - 0.10 / height
        captions = {}
        for ax in row_axes:
            pos = ax.get_position()
            captions[ax] = fig.text(
                (pos.x0 + pos.x1) / 2, caption_top, ax._paper_panel_caption,
                ha="center", va="top", fontsize=11.5, fontweight="normal",
            )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        caption_bottom = min(
            caption.get_window_extent(renderer).transformed(fig.transFigure.inverted()).y0
            for caption in captions.values()
        )
        for ax in row_axes:
            if ax not in legends:
                continue
            handles, labels = legends[ax]
            # 方法较多时分成两列；三种子的短标签放在同一行。
            columns = 2 if len(labels) >= 7 else (3 if all(t.startswith("种子") for t in labels) else 1)
            pos = ax.get_position()
            fig.legend(
                handles, labels, loc="upper center",
                bbox_to_anchor=((pos.x0 + pos.x1) / 2, caption_bottom - 0.10 / height),
                bbox_transform=fig.transFigure, ncol=columns,
                frameon=False, fontsize=9, borderaxespad=0,
                columnspacing=1.1, handletextpad=0.6, labelspacing=0.4,
            )


def save(fig, name, args):
    place_bottom_captions(fig)
    # SVG中的文字默认转为轮廓，避免另一台机器缺少中文字体。
    for ext in args.formats:
        fig.savefig(args.output_dir / f"{name}.{ext}", dpi=args.dpi,
                    bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"[OK] {name}: {', '.join(args.formats)}", flush=True)


def plot3(root, args):
    methods = (*METHODS, "random_k_clip_noise")
    data = {m: training(root, m) for m in methods}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.6, 6.5), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(left=.065, right=.985, top=.95, bottom=.43, wspace=.25)
    panel(ax, "(a) 验证准确率：三个训练种子")
    for method, ls in zip(("fedavg", "random_k", "random_k_clip_noise", "drpe"), ("--", "-.", ":", "-")):
        vals = data[method][0]
        mean, sd = vals.mean(axis=0), vals.std(axis=0, ddof=1)
        rounds = np.arange(1, 201)
        ax.plot(rounds, mean, color=COLORS[method], ls=ls, lw=1.45, label=NAMES[method])
        ax.fill_between(rounds, mean-sd, mean+sd, color=COLORS[method], alpha=.10, linewidth=0)
    ax.set(xlabel="通信轮次", ylabel="验证准确率 / %", xlim=(1, 200), ylim=(10, 95))
    ax.legend(loc="lower right", frameon=False)
    panel(bx, "(b) 上行通信量与测试准确率")
    for i, method in enumerate(methods):
        _, accuracy, uplink = data[method]
        bx.errorbar(uplink.mean(), accuracy.mean(), yerr=accuracy.std(ddof=1),
                    fmt=MARKERS[i], color=COLORS[method], ms=7 if method != "drpe" else 11,
                    capsize=3, elinewidth=.9, label=NAMES[method], zorder=4)
    bx.set(xlabel="每轮上行通信量 / MiB", ylabel="测试准确率 / %", xlim=(20, 93), ylim=(78, 90))
    bx.legend(loc="lower right", frameon=False, fontsize=8.4, labelspacing=.35)
    fig.text(.065, .075, "阴影与误差线：三个训练种子的均值 ± 样本标准差；曲线未经平滑。", fontsize=9, color="#444444")
    fig.text(.065, .035, "Random-k指原无裁剪、无噪声配置；测试准确率取验证集最优检查点。", fontsize=9, color="#444444")
    save(fig, "fig3_utility_communication", args)


def plot4(root, args):
    methods = ("static", "clip_only", "drpe")
    data = {m: training(root, m) for m in methods}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.6, 6.5), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(left=.065, right=.985, top=.95, bottom=.40, wspace=.25)
    panel(ax, "(a) 不同组件配置的训练过程")
    for method, ls in zip(methods, ("--", "-.", "-")):
        vals = data[method][0]; mean = vals.mean(axis=0); sd = vals.std(axis=0, ddof=1)
        ax.plot(np.arange(1, 201), mean, color=COLORS[method], lw=1.4, ls=ls, label=NAMES[method])
        ax.fill_between(np.arange(1, 201), mean-sd, mean+sd, color=COLORS[method], alpha=.11, linewidth=0)
    ax.set(xlabel="通信轮次", ylabel="验证准确率 / %", xlim=(1, 200), ylim=(10, 95))
    ax.legend(loc="lower right", frameon=False)
    panel(bx, "(b) 各训练种子的测试准确率")
    for i, method in enumerate(methods):
        vals = data[method][1]
        bx.errorbar(i, vals.mean(), yerr=vals.std(ddof=1), fmt="_", markersize=21,
                    color=COLORS[method], elinewidth=1.3, capsize=6, zorder=2)
        for offset, val, marker in zip((-.105, 0, .105), vals, ("o", "s", "^")):
            bx.scatter(i+offset, val, marker=marker, s=43, facecolors="white", edgecolors=COLORS[method], zorder=3)
        bx.text(i, vals.max()+.8, f"{vals.mean():.2f} ± {vals.std(ddof=1):.2f}", ha="center", fontsize=10)
    bx.set(xticks=range(3), xticklabels=["固定随机状态\n＋裁剪＋噪声", "动态投影＋裁剪\n无噪声", "完整DRPE"],
           ylabel="测试准确率 / %", xlim=(-.5, 2.5), ylim=(75, 89))
    bx.legend(handles=[Line2D([], [], color="#555555", marker=mark, ls="", markerfacecolor="white", label=f"种子{s}")
                       for s, mark in zip(SEEDS, ("o", "s", "^"))], loc="upper left", frameon=False, ncol=3, fontsize=8.5)
    fig.text(.065, .065, "阴影与误差线：三个种子的样本标准差；各点分别表示一个训练种子。", fontsize=9, color="#444444")
    fig.text(.065, .025, "固定随机状态配置同时固定投影和同一客户端的扰动生成状态。", fontsize=9, color="#444444")
    save(fig, "fig4_component_training", args)


def key(row):
    return (int(row["model_seed"]), row["target_id"], int(row["restart"]))


def validate_attack(rows, description):
    expected = set(itertools.product(SEEDS, [f"T{i:02d}" for i in range(10)], (0, 1)))
    keys = [key(r) for r in rows]
    if len(keys) != 60 or len(set(keys)) != 60 or set(keys) != expected:
        raise ValueError(f"{description}: 需要完整且不重复的3种子×10目标×2初始化，共60条。")
    for field in ("psnr", "ssim", "label_correct"):
        numbers(rows, field)
    if not np.isin(numbers(rows, "label_correct"), [0, 1]).all():
        raise ValueError(f"{description}: label_correct必须是0/1。")
    return sorted(rows, key=key)


def main_attack(root):
    candidates = [root / "privacy_lpips_statistics_results/summaries/per_run_metrics.csv",
                  root / "privacy_main_benchmark/results/summaries/per_run_metrics.csv"]
    for path in candidates:
        if not path.is_file():
            continue
        rows = read_csv(path)
        if all(r.get("lpips", "").strip() for r in rows) and rows:
            break
    else:
        raise FileNotFoundError("未找到补齐LPIPS的主攻击逐次指标表。不能把空值当作0。")
    groups = {}
    for method, attack in itertools.product(METHODS, ("dlg", "idlg")):
        rr = validate_attack([r for r in rows if r["method"] == method and r["attack"] == attack], f"{method}/{attack}")
        numbers(rr, "lpips")
        groups[(method, attack)] = rr
    return groups


def summarize_attack(rows):
    return {f: stat(numbers(rows, f)) for f in ("psnr", "ssim")} | {
        "label_correct": int(numbers(rows, "label_correct").sum()), "n_runs": len(rows), "n_targets": 10,
    }


def plot5(root, args, groups):
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 6.5), gridspec_kw={"width_ratios": [1, 1.08, 1.1]})
    fig.subplots_adjust(left=.055, right=.987, top=.95, bottom=.42, wspace=.29)
    ax, bx, cx = axes
    panel(ax, "(a) iDLG标签恢复率")
    rates = [100*numbers(groups[(m, "idlg")], "label_correct").mean() for m in METHODS]
    bars = ax.bar(range(7), rates, width=.65, color=[COLORS[m] for m in METHODS], edgecolor="white", linewidth=.5)
    for bar, value in zip(bars, rates):
        ax.text(bar.get_x()+bar.get_width()/2, value+2.3, f"{value:.1f}", ha="center", fontsize=8.4)
    ax.set(ylabel="标签恢复率 / %", ylim=(0, 113), xticks=range(7), xticklabels=[NAMES[m] for m in METHODS])
    panel(bx, "(b) iDLG感知距离分布")
    lpips = [numbers(groups[(m, "idlg")], "lpips") for m in METHODS]
    boxes = bx.boxplot(lpips, positions=range(7), widths=.56, patch_artist=True,
                       medianprops={"color": "#202020", "linewidth": 1.1},
                       flierprops={"marker": ".", "markersize": 3, "markeredgecolor": "#777777"})
    for box, m in zip(boxes["boxes"], METHODS):
        box.set(facecolor=COLORS[m], alpha=.55, linewidth=.7)
    bx.set(ylabel="LPIPS", ylim=(0, max(x.max() for x in lpips)*1.08),
           xticks=range(7), xticklabels=[NAMES[m] for m in METHODS])
    panel(cx, "(c) iDLG重建PSNR的经验累积分布")
    styles = ("--", "-", "-.", ":", (0, (4, 1, 1, 1)), (0, (2, 1)), "-")
    for m, ls in zip(METHODS, styles):
        vals = np.sort(numbers(groups[(m, "idlg")], "psnr"))
        xx = np.r_[0., vals]
        yy = np.r_[0., np.arange(1, len(vals)+1)/len(vals)]
        cx.step(xx, yy, where="post", color=COLORS[m], ls=ls, lw=1.9 if m == "drpe" else 1.2, label=NAMES[m])
    cx.set(xlabel="PSNR / dB", ylabel="累计运行比例", xlim=(0, 29), ylim=(0, 1.025))
    cx.legend(loc="upper left", frameon=False, fontsize=8)
    for aa in (ax, bx):
        plt.setp(aa.get_xticklabels(), rotation=38, ha="right", rotation_mode="anchor")
    fig.text(.055, .075, "每种方法60次运行：10个固定目标 × 3个模型种子 × 2次初始化；重复运行不视为独立目标。", fontsize=9, color="#444444")
    fig.text(.055, .035, "箱体：四分位区间；中线：中位数；须：1.5倍四分位距范围内的观测值。Random-k为原无裁剪、无噪声配置。", fontsize=9, color="#444444")
    VALUES["figure5"] = {m: {**summarize_attack(groups[(m, "idlg")]), "lpips": stat(lpips[i])} for i, m in enumerate(METHODS)}
    save(fig, "fig5_idlg_comparison", args)


def raw_attack(root, folder, attack):
    rows = []
    for seed, target, restart in itertools.product(SEEDS, range(10), (0, 1)):
        p = root / folder / f"seed{seed}" / f"T{target:02d}" / f"restart{restart:02d}" / attack / "classic_attack_metrics.csv"
        rr = read_csv(p)
        if len(rr) != 1:
            raise ValueError(f"{p}: 预期只有一个目标指标行。")
        rows.append({**rr[0], "model_seed": seed, "target_id": f"T{target:02d}", "restart": restart})
    return validate_attack(rows, str(folder)+"/"+attack)


def paired_targets(a, b):
    for ra, rb in zip(a, b):
        if key(ra) != key(rb) or int(ra["sample_index"]) != int(rb["sample_index"]) or int(ra["true_label"]) != int(rb["true_label"]):
            raise ValueError("组件比较的目标、标签、模型种子或初始化不匹配。")


def plot6(root, args, groups):
    oracle = {name: raw_attack(root, f"privacy_oracle_benchmark/results/raw/{name}", "oracle")
              for name in ("fedavg_raw", "drpe_encoded", "drpe_decoded")}
    for rows in oracle.values():
        if not all(r["label_source"] == "oracle_known" for r in rows):
            raise ValueError("已知标签分组中出现非oracle记录。")
    modes = ("dlg", "idlg", "oracle")
    no_noise = {m: raw_attack(root, "privacy_component_benchmark/results/clip_only_model_eta0", m) for m in modes}
    full = {"dlg": groups[("drpe", "dlg")], "idlg": groups[("drpe", "idlg")], "oracle": oracle["drpe_encoded"]}
    for mode in modes:
        paired_targets(no_noise[mode], full[mode])
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 10.0))
    fig.subplots_adjust(left=.085, right=.98, bottom=.24, top=.95, hspace=.75, wspace=.27)
    names = ("FedAvg\n原始梯度", "DRPE\n编码表示", "DRPE\n解码表示")
    for j, field in enumerate(("psnr", "ssim")):
        ax = axes[0, j]
        panel(ax, f"({'ab'[j]}) 已知真实标签：{'PSNR' if field=='psnr' else 'SSIM'}")
        for i, rows in enumerate(oracle.values()):
            vals = numbers(rows, field)
            col = (COLORS["fedavg"], COLORS["drpe"], COLORS["dense_rp"])[i]
            ax.bar(i, vals.mean(), .55, color=col, alpha=.8, zorder=3)
            ax.errorbar(i, vals.mean(), yerr=vals.std(ddof=1), fmt="none", color="#333333", capsize=4, lw=.9, zorder=4)
            ax.text(i, vals.mean()+vals.std(ddof=1)+(0.7 if field=="psnr" else .025),
                    f"{vals.mean():.2f}" if field=="psnr" else f"{vals.mean():.3f}", ha="center", fontsize=9.5)
        ax.set(xticks=range(3), xticklabels=names, ylabel="PSNR / dB" if field=="psnr" else "SSIM",
               ylim=(0, 26) if field=="psnr" else (0, .9))
    for j, field in enumerate(("psnr", "ssim")):
        ax = axes[1, j]
        panel(ax, f"({'cd'[j]}) 组件配置比较：{'PSNR' if field=='psnr' else 'SSIM'}")
        for offset, dataset, color, label, hatch in [(-.17, no_noise, COLORS["clip_only"], "动态投影＋裁剪，无噪声", ""),
                                                    (.17, full, COLORS["drpe"], "完整DRPE", "//")]:
            means = [numbers(dataset[m], field).mean() for m in modes]
            sds = [numbers(dataset[m], field).std(ddof=1) for m in modes]
            ax.bar(np.arange(3)+offset, means, .30, color=color, alpha=.8, label=label, hatch=hatch,
                   edgecolor="white", linewidth=.5, zorder=3)
            ax.errorbar(np.arange(3)+offset, means, yerr=sds, fmt="none", color="#333333", capsize=3, lw=.9, zorder=4)
        ax.set(xticks=range(3), xticklabels=["DLG", "iDLG", "已知标签\n编码空间"],
               ylabel="PSNR / dB" if field=="psnr" else "SSIM", ylim=(0, 30) if field=="psnr" else (0, 1.12))
        if j == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(0, 1.01), frameon=False, fontsize=8)
    fig.text(.085, .054, "误差线：60次运行的样本标准差，并非置信区间；每组仅包含10个不同目标。", fontsize=9, color="#444444")
    fig.text(.085, .021, "组件配置使用各自训练的模型检查点；此图不将差异解释为固定模型上噪声的独立作用。", fontsize=9, color="#444444")
    VALUES["figure6"] = {"oracle": {m: summarize_attack(r) for m, r in oracle.items()},
                         "components": {m: {"no_noise": summarize_attack(no_noise[m]), "full_drpe": summarize_attack(full[m])} for m in modes}}
    save(fig, "fig6_label_noise_comparison", args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="实验项目根目录")
    parser.add_argument("--output-dir", type=Path, default=Path("figures/revision_fig3_6"))
    parser.add_argument("--figures", nargs="+", type=int, choices=(3, 4, 5, 6), default=[3, 4, 5, 6])
    parser.add_argument("--formats", nargs="+", choices=("svg", "pdf", "png"), default=["svg", "pdf", "png"])
    parser.add_argument("--dpi", type=int, default=600, help="仅影响PNG分辨率，矢量格式不受影响")
    parser.add_argument("--font", help="自定义中文字体文件路径")
    args = parser.parse_args()
    args.root = args.root.resolve()
    if not args.output_dir.is_absolute():
        args.output_dir = args.root / args.output_dir
    args.output_dir = args.output_dir.resolve()
    if not (args.root / "outputs").is_dir():
        parser.error("--root必须指向drpe_formal_experiments项目根目录。")
    # 避免输出目录指向已有原始结果。
    raw_roots = [args.root / p for p in ("outputs", "privacy_main_benchmark", "privacy_oracle_benchmark",
                                        "privacy_component_benchmark", "privacy_lpips_statistics_results")]
    if any(args.output_dir == p or p in args.output_dir.parents for p in raw_roots):
        parser.error("--output-dir不能位于原始实验结果目录内，请使用figures等独立目录。")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_style(args.font)
    if 3 in args.figures:
        plot3(args.root, args)
    if 4 in args.figures:
        plot4(args.root, args)
    if 5 in args.figures or 6 in args.figures:
        groups = main_attack(args.root)
        if 5 in args.figures:
            plot5(args.root, args, groups)
        if 6 in args.figures:
            plot6(args.root, args, groups)
    manifest = {"project_root": str(args.root), "figures": args.figures,
                "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                "sources_sha256": SOURCES,
                "statistics": "sample standard deviation ddof=1; no smoothing; attack runs are repeated measurements"}
    (args.output_dir / "plot_source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "plot_values.json").write_text(json.dumps(VALUES, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"输出目录：{args.output_dir}")


if __name__ == "__main__":
    main()
