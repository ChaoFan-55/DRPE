$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$configs = @(
  "configs/cifar100/resnet18/attacks/fedavg/dlg_raw_best_seed42.json",
  "configs/cifar100/resnet18/attacks/fedavg/idlg_raw_best_seed42.json",
  "configs/cifar100/resnet18/attacks/drpe/dlg_encoded_best_seed42.json",
  "configs/cifar100/resnet18/attacks/drpe/idlg_decoded_best_seed42.json",
  "configs/cifar100/resnet18/attacks/fedavg/dlg_raw_best_seed43.json",
  "configs/cifar100/resnet18/attacks/fedavg/idlg_raw_best_seed43.json",
  "configs/cifar100/resnet18/attacks/drpe/dlg_encoded_best_seed43.json",
  "configs/cifar100/resnet18/attacks/drpe/idlg_decoded_best_seed43.json",
  "configs/cifar100/resnet18/attacks/fedavg/dlg_raw_best_seed44.json",
  "configs/cifar100/resnet18/attacks/fedavg/idlg_raw_best_seed44.json",
  "configs/cifar100/resnet18/attacks/drpe/dlg_encoded_best_seed44.json",
  "configs/cifar100/resnet18/attacks/drpe/idlg_decoded_best_seed44.json",
)
foreach ($config in $configs) {
  Write-Host "Running $config"
  python scripts/run_classic_attack_experiment.py --config $config
  if ($LASTEXITCODE -ne 0) { throw "Failed: $config" }
}
