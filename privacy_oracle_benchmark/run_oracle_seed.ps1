param(
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Variants = @(
    "fedavg_raw",
    "drpe_encoded",
    "drpe_decoded"
)

Write-Host "===================================================================="
Write-Host "Oracle-label benchmark"
Write-Host "Seed = $Seed"
Write-Host "===================================================================="

foreach ($Variant in $Variants) {

    $ConfigDir = Join-Path `
        $ProjectRoot `
        "privacy_oracle_benchmark\results\generated_configs\$Variant"

    if (-not (Test-Path $ConfigDir)) {
        throw "Missing config directory: $ConfigDir"
    }

    $Pattern = "*__m${Seed}__T*__oracle__r*.json"

    $Configs = Get-ChildItem `
        -Path $ConfigDir `
        -Filter $Pattern `
        | Sort-Object Name

    Write-Host ""
    Write-Host "--------------------------------------------------------------------"
    Write-Host "Variant: $Variant"
    Write-Host "Configs found: $($Configs.Count)"
    Write-Host "--------------------------------------------------------------------"

    if ($Configs.Count -ne 20) {
        throw "Expected 20 configs for $Variant seed $Seed, found $($Configs.Count)"
    }

    foreach ($Config in $Configs) {

        $Json = Get-Content $Config.FullName -Raw | ConvertFrom-Json
        $SummaryPath = Join-Path `
            $Json.output_dir `
            "classic_attack_summary.json"

        if (Test-Path $SummaryPath) {
            Write-Host "[SKIP] Already completed:"
            Write-Host $Config.Name
            continue
        }

        Write-Host ""
        Write-Host "===================================================================="
        Write-Host "[RUN] $($Config.Name)"
        Write-Host "===================================================================="

        python .\scripts\run_classic_attack_experiment.py `
            --config $Config.FullName

        if ($LASTEXITCODE -ne 0) {
            throw "Attack failed: $($Config.FullName)"
        }
    }
}

Write-Host ""
Write-Host "===================================================================="
Write-Host "Seed $Seed oracle-label benchmark completed."
Write-Host "===================================================================="