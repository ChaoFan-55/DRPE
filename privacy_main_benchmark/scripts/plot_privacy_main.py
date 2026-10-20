from __future__ import annotations
import argparse, csv
from pathlib import Path
import matplotlib.pyplot as plt

BENCH = Path(__file__).resolve().parents[1]


def read_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def grouped_plot(rows, metric, ylabel, filename):
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])
    attacks = ["dlg", "idlg"]
    x = list(range(len(methods)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for j, attack in enumerate(attacks):
        values = []
        errors = []
        for method in methods:
            found = next((r for r in rows if r["method"] == method and r["attack"] == attack), None)
            values.append(float(found[f"{metric}_mean"]) if found and found.get(f"{metric}_mean") not in (None, "", "None") else 0.0)
            errors.append(float(found[f"{metric}_std"]) if found and found.get(f"{metric}_std") not in (None, "", "None") else 0.0)
        xpos = [v + (j - 0.5) * width for v in x]
        ax.bar(xpos, values, width=width, yerr=errors, capsize=3, label=attack.upper())
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel(ylabel); ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    out = BENCH / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(out / f"{filename}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(BENCH / "results" / "summaries" / "privacy_summary_by_method_attack.csv"))
    args = parser.parse_args()
    rows = read_rows(args.summary)
    grouped_plot(rows, "psnr", "Attack PSNR (dB) ↑", "privacy_psnr")
    grouped_plot(rows, "ssim", "Attack SSIM ↑", "privacy_ssim")
    print(f"Saved figures to: {BENCH / 'results' / 'figures'}")

if __name__ == "__main__":
    main()
