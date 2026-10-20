param([string]$Device = "auto")
$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$projectRoot = Split-Path -Parent $PSScriptRoot
foreach ($seed in @(42, 43, 44)) {
    $trainArgs = @("-B", "$PSScriptRoot/train.py", "--project-root", $projectRoot, "--seed", "$seed", "--device", $Device)
    if (Test-Path -LiteralPath "$PSScriptRoot/results/formal/seed$seed/resume.pt") { $trainArgs += "--resume" }
    & python @trainArgs
    if ($LASTEXITCODE -ne 0) { throw "Seed $seed training failed." }
}
