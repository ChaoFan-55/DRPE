param(
    [ValidateSet(42, 43, 44)]
    [int]$ModelSeed = 42,

    [ValidateSet("clip_only", "full_drpe")]
    [string]$ModelSource = "clip_only"
)

$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

$experimentRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $experimentRoot

if ($ModelSource -eq "clip_only") {
    $modelDir = Join-Path $experimentRoot `
        "outputs\ablation_components\projection_clip_only\seed$ModelSeed"
    $variant = "clip_only_model_eta0"
}
else {
    $modelDir = Join-Path $experimentRoot `
        "outputs\cifar10\resnet18\drpe\seed$ModelSeed"
    $variant = "full_drpe_model_eta0"
}

$checkpointPath = Join-Path $modelDir "checkpoint_round_0020.pt"

if (-not (Test-Path -LiteralPath $checkpointPath)) {
    throw "Missing checkpoint: $checkpointPath"
}

$sourceConfigDir = Join-Path $experimentRoot `
    "privacy_main_benchmark\results\generated_configs\drpe"

$resultRoot = Join-Path $experimentRoot `
    "privacy_component_benchmark\results\$variant"

foreach ($attackMode in @("dlg", "idlg", "oracle")) {
    $sourceAttack = "dlg"
    $observationMode = "encoded"
    $labelMode = "optimize"

    if ($attackMode -eq "idlg") {
        $sourceAttack = "idlg"
        $observationMode = "decoded"
    }
    elseif ($attackMode -eq "oracle") {
        $labelMode = "known"
    }

    foreach ($targetNumber in 0..9) {
        $targetId = "T{0:D2}" -f $targetNumber

        foreach ($restartNumber in 0..1) {
            $restartId = "{0:D2}" -f $restartNumber

            $configName = "drpe__m${ModelSeed}__${targetId}__${sourceAttack}__r${restartId}.json"
            $configPath = Join-Path $sourceConfigDir $configName

            if (-not (Test-Path -LiteralPath $configPath)) {
                throw "Missing source config: $configPath"
            }

            $outputPath = Join-Path $resultRoot `
                "seed$ModelSeed\$targetId\restart$restartId\$attackMode"

            $summaryPath = Join-Path $outputPath "classic_attack_summary.json"
            $metricsPath = Join-Path $outputPath "classic_attack_metrics.csv"

            if ((Test-Path -LiteralPath $summaryPath) -and
                (Test-Path -LiteralPath $metricsPath)) {
                Write-Host "SKIP completed: $outputPath"
                continue
            }

            $attackArgs = @(
                "scripts/run_classic_attack_experiment.py"
                "--config"
                $configPath
                "--set"
                "output_dir=$outputPath"
                "checkpoint_path=$checkpointPath"
                "training_summary_path=null"
                "projection_round=21"
                "dynamic_projection=true"
                "attack_name=$sourceAttack"
                "label_mode=$labelMode"
                "observation_mode=$observationMode"
                "observation_clip_norm=2.950101638259676"
                "noise_eta=0.0"
                "noise_std=0.0"
                "iterations=2000"
                "compute_optional_lpips=true"
            )

            Write-Host "RUN: $variant / seed$ModelSeed / $targetId / restart$restartId / $attackMode"

            & python @attackArgs

            if ($LASTEXITCODE -ne 0) {
                throw "Attack failed: $outputPath"
            }

            if (-not (Test-Path -LiteralPath $summaryPath)) {
                throw "Missing completed summary: $outputPath"
            }
        }
    }
}

Write-Host "Completed: $variant / seed$ModelSeed"o