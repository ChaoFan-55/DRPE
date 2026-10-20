param(
    [ValidateSet("auto", "cuda", "cpu")][string]$Device = "cuda",
    [string]$PythonExe = "python",
    [string]$ProjectRoot = "",
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable not found: $PythonExe. Set -PythonExe to your experiment environment."
}
if ($ProjectRoot -eq "") { $ProjectRoot = Split-Path -Parent $PSScriptRoot }
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$runnerArgs = @("-u", "-B", "$PSScriptRoot/run_overnight.py", "--project-root", $ProjectRoot, "--device", $Device)
if ($DryRun) { $runnerArgs += "--dry-run" }
& $PythonExe @runnerArgs
if ($LASTEXITCODE -ne 0) { throw "Overnight runner stopped. See pri_prune_baseline/results/overnight." }
