$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
python scripts/run_calibration.py --config configs/cifar100/resnet18/calibration/calibrate_p75_200r.json
