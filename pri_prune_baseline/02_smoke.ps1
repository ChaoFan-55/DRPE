param([string]$Device = "auto")
$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$projectRoot = Split-Path -Parent $PSScriptRoot
$trainArgs = @("-B", "$PSScriptRoot/train.py", "--project-root", $projectRoot, "--seed", "42", "--device", $Device, "--smoke")
if (Test-Path -LiteralPath "$PSScriptRoot/results/smoke/seed42/resume.pt") { $trainArgs += "--resume" }
& python @trainArgs
if ($LASTEXITCODE -ne 0) { throw "Smoke training failed." }
foreach ($attack in @("dlg", "idlg")) {
    & python -B "$PSScriptRoot/attack.py" --project-root $projectRoot --seed 42 --target T00 --restart 0 --attack $attack --device $Device --smoke --skip-lpips
    if ($LASTEXITCODE -ne 0) { throw "Smoke attack failed." }
}
