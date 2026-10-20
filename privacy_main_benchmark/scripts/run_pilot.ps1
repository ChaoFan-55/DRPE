$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
python .\privacy_main_benchmark\scripts\build_manifest.py --mode pilot
python .\privacy_main_benchmark\scripts\run_privacy_pilot.py
