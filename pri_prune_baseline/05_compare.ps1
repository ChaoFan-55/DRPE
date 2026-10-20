$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"
$projectRoot = Split-Path -Parent $PSScriptRoot
& python -B "$PSScriptRoot/compare.py" --project-root $projectRoot
if ($LASTEXITCODE -ne 0) { throw "Comparison failed; check missing or incompatible runs." }
