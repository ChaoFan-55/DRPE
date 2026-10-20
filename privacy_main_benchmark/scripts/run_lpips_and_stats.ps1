$ErrorActionPreference = "Stop"

Write-Host "[1/4] Checking LPIPS package..."
python -c "import lpips" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "LPIPS is not installed. Installing..."
    python -m pip install lpips
    if ($LASTEXITCODE -ne 0) { throw "Failed to install lpips." }
}

Write-Host "[2/4] Offline LPIPS for all 840 runs..."
python .\privacy_main_benchmark\scripts\compute_lpips_offline.py --update-per-run
if ($LASTEXITCODE -ne 0) { throw "Offline LPIPS failed." }

Write-Host "[3/4] Target-aware statistics and publication figures..."
python .\privacy_main_benchmark\scripts\analyze_privacy_main_v2.py
if ($LASTEXITCODE -ne 0) { throw "Statistics/plotting failed." }

Write-Host "[4/4] Verifying LPIPS coverage..."
python -c "import pandas as pd; p=r'.\privacy_main_benchmark\results\summaries\per_run_metrics.csv'; d=pd.read_csv(p); n=d['lpips'].notna().sum(); print(f'LPIPS coverage: {n}/{len(d)}'); assert n==len(d)"
if ($LASTEXITCODE -ne 0) { throw "LPIPS coverage is incomplete." }

Write-Host "Done."
Write-Host "Tables:  .\privacy_main_benchmark\results\statistics_v2\tables"
Write-Host "Figures: .\privacy_main_benchmark\results\statistics_v2\figures"
