$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

python .\projection_efficiency_benchmark\benchmark_projection_efficiency.py `
  --config .\projection_efficiency_benchmark\config.json

if ($LASTEXITCODE -ne 0) {
    throw "Projection efficiency benchmark failed."
}

Write-Host ""
Write-Host "Projection efficiency benchmark completed."
