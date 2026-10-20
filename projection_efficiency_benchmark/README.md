# DRPE vs Dense-RP Projection Efficiency Benchmark

This is the final system-efficiency supplement for the DRPE paper.

## What it measures

At the formal projection setting (ratio=0.3, block size=512):

1. Projection-state construction time per block.
2. Cached encode time.
3. Cached decode time.
4. Cached encode+decode time.
5. Dynamic state-construction + encode time.
6. Dynamic state-construction + encode+decode time.
7. Peak extra CUDA memory for a full block operation.
8. Projection-state storage per block.
9. Exact model-wide explicit dense-matrix storage implied by the existing
   ResNet-18 parameter shapes and tensor-by-tensor blocking.

The benchmark uses three projection seeds: 2026/2027/2028 and 200 timed repeats per seed.

## Fairness / interpretation

Dense-RP is implemented as a block-wise dense row-orthonormal Gaussian projection.
For an n-dimensional block and m retained coordinates, an n x m Gaussian matrix is
QR-factorized and transposed to obtain an m x n matrix with orthonormal rows.

DRPE uses the project's actual normalized FWHT kernel and a sign vector plus sampled
indices, corresponding to P = S H D.

Two timing regimes are reported:
- cached: projection state for the current round already exists;
- dynamic: state construction is included.

Do NOT write that Dense-RP necessarily stores all model projection matrices in a
real deployment. The model-wide dense-matrix storage number is the footprint
*if explicit matrices are cached*. A memory-conscious implementation can regenerate
matrices block-by-block, trading storage for repeated construction cost.

DRPE's strongest structural claim is matrix-free representation: it requires no
explicit dense projection matrix.

## Run

From the project root:

```powershell
conda activate PTF-FSR
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"

python .\projection_efficiency_benchmark\benchmark_projection_efficiency.py `
  --config .\projection_efficiency_benchmark\config.json
```

or:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\projection_efficiency_benchmark\run_benchmark.ps1
```

## Expected output files

`projection_efficiency_benchmark/results/`

- benchmark_seed_results.csv
- benchmark_summary.csv
- projection_storage_summary.json
- resolved_benchmark_config.json

Upload these four files after the run.
