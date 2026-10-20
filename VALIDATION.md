# Verification performed during packaging

Prepared on 2026-10-20 from existing experiment files. No full training runs or 2000-step inversion experiments were newly executed for this release.

## Numerical records

- All 45 included formal training runs contain 200 per-round records.
- Main attack aggregation contains 960 runs: the seven original methods/DRPE produce 840, and PriPrune-Magnitude contributes 120.
- Known-label and no-noise configuration evaluations each contain 180 runs.
- The 960 rows in the final English figure main-attack snapshot match the archived per-run PSNR, SSIM, LPIPS, and label-correctness records.
- The 360 rows in the additional attack figure snapshot match the original known-label/component records.
- All 33 main/ablation training summary rows used for the final English figures match the original accuracy and mean per-round uplink values.
- Tables 2--9 were regenerated. Key manuscript values, including DRPE accuracy 84.23%, PriPrune-Magnitude accuracy 81.99%, and DRPE iDLG PSNR 13.15 dB, SSIM 0.364, and label recovery 56.7%, were checked.

Run `python release_tools/summarize_paper.py` to repeat the aggregation checks. See `paper_data/derived/verification.json`.

## Code and configuration

- All 45 generated reproduction configurations were parsed by the corresponding original configuration loaders in the experiment's Python environment.
- All 90 Python source files in the release were parsed for syntax errors.
- Dry-run commands were checked for training and for original main, PriPrune, known-label, and no-noise attack entry points. No expensive experiments were launched.
- Selected existing tests of projection, weighted aggregation, PriPrune masking/comparison, and the public-support attack adapter passed. The first invocation had 12 passed and one conditional adapter test skipped because its project-root environment variable was unset; that test was then run with the release root explicitly supplied and passed.
- The two original six-combination configuration-matrix tests are omitted because the release intentionally includes only the three combinations reported in this manuscript. Algorithm implementations are preserved.

## File integrity

- Copied source files have original and release SHA-256 values in `source_file_manifest.csv`. Recorded transformations only make local paths portable.
- All 27 model checkpoints in the separate checkpoint ZIP were read back from the compressed archive and verified against their original SHA-256 values.
- Raw algorithm implementations, metric values, images, and checkpoint bytes are preserved. The added release helpers, generated summary tables, reproduction configurations, and documentation are separately identifiable.
- The original experiment directory was used as a read-only source. All preparation outputs, Python test temporaries, and archives were written under the Codex workspace.

This verification supports artifact completeness and consistency with the saved records; it does not establish bitwise reproduction on a different GPU, software version, or operating system.
