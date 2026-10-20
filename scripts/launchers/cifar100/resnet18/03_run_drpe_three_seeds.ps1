$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$configs = @(
  "configs/cifar100/resnet18/drpe/drpe_r030_p75_eta200_seed42_fixed200.json",
  "configs/cifar100/resnet18/drpe/drpe_r030_p75_eta200_seed43_fixed200.json",
  "configs/cifar100/resnet18/drpe/drpe_r030_p75_eta200_seed44_fixed200.json",
)
foreach ($config in $configs) {
  Write-Host "Running $config"
  python scripts/run_experiment.py --config $config
  if ($LASTEXITCODE -ne 0) { throw "Failed: $config" }
}
