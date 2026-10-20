param([string]$Device = "auto")
$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$projectRoot = Split-Path -Parent $PSScriptRoot
& python -B "$PSScriptRoot/preflight.py" --project-root $projectRoot --check-data
if ($LASTEXITCODE -ne 0) { throw "Preflight failed; do not start formal runs." }
