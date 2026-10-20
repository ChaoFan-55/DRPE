from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_one(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize FLoCoRA round-20 DLG/iDLG attacks.")
    parser.add_argument(
        "--root",
        default="./outputs/cifar10/resnet18/attacks/flocora",
        help="Root directory containing seed42/seed43/seed44 attack outputs.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    rows: list[dict] = []
    for seed in (42, 43, 44):
        for attack in ("dlg", "idlg"):
            path = (
                root
                / f"seed{seed}"
                / f"{attack}_round20"
                / "flocora_attack_factor_grad_summary.json"
            )
            if not path.exists():
                print(f"[MISS] {path}")
                continue
            result = load_one(path)
            rows.append(
                {
                    "seed": seed,
                    "attack": attack.upper(),
                    "psnr": float(result["psnr"]),
                    "ssim": float(result["ssim"]),
                    "lpips": result.get("lpips"),
                    "label_correct": int(result["label_correct"]),
                    "predicted_label": int(result["predicted_label"]),
                    "true_label": int(result["true_label"]),
                    "best_iteration": int(result["best_iteration"]),
                    "best_match_loss": float(result["best_match_loss"]),
                }
            )

    if not rows:
        raise SystemExit("No FLoCoRA attack summaries were found.")

    print("\nPer-run results")
    print("seed  attack  PSNR(dB)   SSIM     label")
    print("----  ------  --------   ------   -----")
    for row in rows:
        print(
            f"{row['seed']:>4}  {row['attack']:<6}  {row['psnr']:>8.2f}   "
            f"{row['ssim']:>6.4f}   {row['predicted_label']}/{row['true_label']}"
        )

    print("\nMean ± sample std across seeds")
    for attack in ("DLG", "IDLG"):
        subset = [row for row in rows if row["attack"] == attack]
        if not subset:
            continue
        psnr = np.array([row["psnr"] for row in subset], dtype=float)
        ssim = np.array([row["ssim"] for row in subset], dtype=float)
        labels = sum(row["label_correct"] for row in subset)
        ddof = 1 if len(subset) > 1 else 0
        print(
            f"{attack}: PSNR={psnr.mean():.2f}±{psnr.std(ddof=ddof):.2f} dB, "
            f"SSIM={ssim.mean():.4f}±{ssim.std(ddof=ddof):.4f}, "
            f"label={labels}/{len(subset)}"
        )

    output = root / "summary_round20_factor.json"
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {output.resolve()}")


if __name__ == "__main__":
    main()
