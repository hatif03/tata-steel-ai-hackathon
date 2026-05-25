# Create and bootstrap the project virtual environment (Windows PowerShell).
# Usage: powershell -ExecutionPolicy Bypass -File scripts/setup.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment at .venv ..."
    python -m venv .venv
}

Write-Host "Installing dependencies ..."
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt

Write-Host ""
Write-Host "Done. Activate with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then train:"
Write-Host "  python models/xgboost-baseline/train.py"
