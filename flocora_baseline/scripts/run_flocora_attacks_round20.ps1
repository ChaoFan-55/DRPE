$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

# Run from project root:
# .
#
# Primary protocol:
#   checkpoint = round 20
#   simulated transmission = round 21
#   sample = 1
#   observation = factor (LoRA A/B + GroupNorm + final FC trainable gradients)
#   DLG + iDLG, 2000 iterations

$jobs = @(
    @{ Seed = 42; Checkpoint = "./outputs/cifar10/resnet18/flocora/commmatched_seed42/checkpoint_round_0020.pt"; Tag = "seed42" },
    @{ Seed = 43; Checkpoint = "./outputs/cifar10/resnet18/flocora/commmatched_seed43/checkpoint_round_0020.pt"; Tag = "seed43" },
    @{ Seed = 44; Checkpoint = "./outputs/cifar10/resnet18/flocora/commmatched_seed44/checkpoint_round_0020.pt"; Tag = "seed44" }
)

foreach ($job in $jobs) {
    foreach ($attack in @("dlg", "idlg")) {
        $outDir = "./outputs/cifar10/resnet18/attacks/flocora/$($job.Tag)/${attack}_round20"
        Write-Host "================================================================================"
        Write-Host "FLoCoRA attack | seed=$($job.Seed) | attack=$attack | output=$outDir"
        Write-Host "================================================================================"

        python ./flocora_baseline/run_flocora_attack.py `
          --checkpoint $job.Checkpoint `
          --output-dir $outDir `
          --attack $attack `
          --observation factor `
          --seed $job.Seed `
          --round-index 21 `
          --sample-index 1 `
          --rank 93 `
          --lora-alpha 1488 `
          --virtual-local-lr 0.01 `
          --iterations 2000 `
          --attack-lr 0.1 `
          --label-lr 0.05 `
          --tv-weight 1e-4 `
          --label-entropy-weight 1e-3 `
          --print-every 25
    }
}

python ./flocora_baseline/scripts/summarize_flocora_attacks.py `
  --root ./outputs/cifar10/resnet18/attacks/flocora
