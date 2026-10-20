# DRPE research materials

Code and recorded experimental results supporting **A Federated Learning Dynamic Random Projection Encoding Method for Communication Compression and Gradient Inversion Defense**.

Authors: Weiping Peng, Siyuan Fan, Cheng Song, Yuan Ping, and Yiran Hu.

This package contains the experiments reported in the manuscript: communication-matched training, random-state and noise configurations, single-sample gradient inversion, known-label controls, cross-model/dataset training, and projection microbenchmarks. PriPrune-Magnitude is an independent adaptation to this protocol, not a verified reproduction of the original paper's native setting.

## Contents

- `src/`, baseline directories, and benchmark directories: the existing experiment implementations.
- `training_runs.csv`: the 45 reported training runs and their original record directories.
- `outputs/` and `pri_prune_baseline/results/formal/`: per-round metrics, final summaries, configurations, partition summaries, and recorded environment information. Model binaries are distributed separately.
- Attack result directories: 960 main DLG/iDLG runs, 180 known-label runs, and 180 no-noise configuration runs; metrics, resolved configurations, original target/reconstruction PNGs, and available progress/command records are retained.
- `paper_data/derived/`: machine-readable counterparts of Tables 2--9, rebuilt from the recorded runs.
- `paper_figures/figure_sources/`: English figure code and the exact numerical snapshots for Figures 3--6, including PriPrune-Magnitude. Figures 1--2 are method schematics and have no numerical experiment data.
- `reproduction_configs/`, `release_tools/`: fresh-output training configs and convenience tools. Original experiment algorithms were not edited during packaging.
- `checkpoint_manifest.csv`: filenames, sizes, and SHA-256 hashes of the 27 round-20 model files used for the attack evaluations.
- `source_file_manifest.csv`: source and release hashes, with path normalization recorded explicitly.

Only the three reported dataset/model combinations are included. Exploratory RCBA/local-update experiments, smoke/diagnostic results, duplicate analysis bundles, original dataset downloads, caches, and unrelated model checkpoints are excluded.

## Rebuild the reported tables without training

From this directory:

```text
python release_tools/summarize_paper.py
```

This uses only the Python standard library. It checks the final figure snapshots against the original per-run records and produces `paper_data/derived/verification.json`. It performs no training and no gradient inversion.

To regenerate English Figures 3--6:

```text
python paper_figures/figure_sources/plot_figures.py
```

This requires NumPy and Matplotlib, uses the supplied figure snapshots, and writes figures into `paper_figures/`. It is the English plotting entry point for this release. Older plotting scripts retained under `scripts/` are historical experiment tools and may omit the subsequently integrated PriPrune-Magnitude comparison.

## Environment

Historical run records identify Python 3.10.20, PyTorch 1.13.1+cu116, CUDA 11.6, and an NVIDIA GeForce RTX 3050 Ti Laptop GPU. `requirements.snapshot.txt` records selected packages read from the experiment environment when the materials were prepared on 2026-10-20; versions other than those recorded in each historical run are a packaging-time snapshot, not a claim about every past run.

Use Python 3.10 in a separate environment:

```text
python -m pip install -r requirements.snapshot.txt
```

Use a compatible PyTorch build for the local platform/hardware if the CUDA 11.6 build cannot be installed. Such a change is a new execution environment and may affect reproducibility. The original `requirements.txt` is retained as `requirements.original.txt`; its `torch>=2.1` declaration does not describe the historical runs and should not be used to upgrade the recorded environment.

## Public datasets

CIFAR-10 and CIFAR-100 are obtained from their original distribution: <https://www.cs.toronto.edu/~kriz/cifar.html>. Complete dataset downloads are not redistributed here. The project dataset loader can download them to `data/`; the included attack target indices identify the exact ten training examples. Some attack entry points, notably the PriPrune adaptation, require the datasets to have been downloaded first.

The source code in `src/data.py`, recorded split seeds, and partition summaries specify the train/validation/client partitions. The Dirichlet split uses its recorded seed offset. Do not replace these with a different random split when comparing runs.

## Retrain one reported configuration

```text
python release_tools/run_training.py --list
python release_tools/run_training.py --run-id cifar10_resnet18_drpe_s42
python release_tools/run_training.py --run-id cifar10_resnet18_drpe_s42 --device cuda --execute
```

The second command is a dry run; only `--execute` starts training. Reproduction outputs go to `reproduced_training/`, never into the archived result directories. The PriPrune helper first makes an isolated copy of its code because its original entry point writes to package-relative result paths. For Random-k+clipping+noise, the reproduction config sets `noise_std=null` so the original entry point recomputes the saved scale using its actual transmitted dimension; this does not alter the archived resolved config or results.

Full 200-round training and 2000-step attacks are computationally expensive. This release was verified by aggregation, file integrity checks, configuration parsing, and selected existing unit tests; it does not claim that all expensive experiments were freshly rerun during packaging.

## Re-evaluate attacks with the archived round-20 models

Download the separate `DRPE_Attack_Checkpoints.zip` and extract it **into this repository directory**. The archive contains `outputs/...` and `pri_prune_baseline/...` paths matching `checkpoint_manifest.csv`.

```text
python release_tools/replay_attack.py --suite main --method drpe --model-seed 42 --target T00 --attack idlg --restart 0
```

The command prints the invocation. Add `--execute` to actually run it. New outputs go to `reproduced_attacks/`, preserving all archived attacks. Other supported suites are `known-label` (methods `fedavg_raw`, `drpe_encoded`, `drpe_decoded`, attack `oracle`) and `no-noise` (method `clip_only`, attacks `dlg`, `idlg`, `oracle`). Main methods include the seven original baselines/DRPE and `pri_prune_magnitude`.

LPIPS in the main comparison was calculated from saved target/reconstruction PNGs with the AlexNet LPIPS model on inputs scaled to [-1,1]. Retain the supplied offline LPIPS CSVs when reproducing the published statistics. A new attack may require the LPIPS/AlexNet weights to download before execution.

## Interpretation of results

- Uploaded training objects are model updates. The inversion benchmark uses single-sample gradient proxies, not the actual updates after five local epochs.
- Projection states are public; projection seeds are not secret keys. The noise scale is empirical and is not a differential-privacy guarantee.
- Each method/attack has 60 runs from 10 targets, three trained model seeds, and two initializations. The 60 runs are not 60 independent targets.
- FLoCoRA attacks use cosine decay of the image-optimizer learning rate to 0.001; the other main attacks use fixed learning rates, as in the saved logs and manuscript.
- The PriPrune-Magnitude attack uses the complete effective local model and transmitted support, as disclosed in the manuscript.
- Projection timing is a single-block GPU microbenchmark. It does not establish an end-to-end speedup; the main training runs project on CPU.
- Cross-dataset/model results evaluate communication and classification, not attack robustness on every task.

## Provenance, citation, and release

Local experiment paths in metadata have been made relative, and the local Python executable path has been replaced with `python` where applicable. Numeric metrics, image pixels, algorithm source code, and model checkpoint bytes are preserved. See `source_file_manifest.csv` for original/release hashes.

`docs/original_project_README.md` and baseline-specific notes retain historical instructions. Where they differ from the final recorded configurations, use the records and this release README. In particular, the early generic README's calibration/attack description is not the final paper protocol.

Before a public release, the authors should confirm software/data reuse licenses, assign a release version, and add the public GitHub URL and archive DOI to the manuscript and citation metadata. No reuse license has been assigned by this packaging step; see `LICENSING.md`. The files are prepared locally and have not been uploaded or made public.
