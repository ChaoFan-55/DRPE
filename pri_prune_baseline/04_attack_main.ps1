param([string]$Device = "auto")
$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$projectRoot = Split-Path -Parent $PSScriptRoot
& python -B "$PSScriptRoot/run_attacks.py" --project-root $projectRoot --device $Device
if ($LASTEXITCODE -ne 0) { throw "Attack batch interrupted. Rerun to skip completed tasks." }
