$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

python .\flocora_baseline\run_flocora.py `
  --config .\flocora_baseline\configs\cifar10_resnet18\flocora_commmatched_seed42_fixed200.json

python .\flocora_baseline\run_flocora.py `
  --config .\flocora_baseline\configs\cifar10_resnet18\flocora_commmatched_seed43_fixed200.json

python .\flocora_baseline\run_flocora.py `
  --config .\flocora_baseline\configs\cifar10_resnet18\flocora_commmatched_seed44_fixed200.json
