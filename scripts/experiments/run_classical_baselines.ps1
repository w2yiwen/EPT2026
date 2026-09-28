[CmdletBinding()]
param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Config = "configs\eptnet_v6_marlin11_4060_windowed_seed42.yaml",
    [string]$Output = "results\baselines\marlin11_classical_seed42.json",
    [string]$Device = "cpu"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot
$env:PYTHONPATH = (Resolve-Path ".\src").Path

& $Python scripts\experiments\run_classical_baselines.py `
    --config $Config `
    --output $Output `
    --device $Device
if ($LASTEXITCODE -ne 0) {
    throw "Classical baselines failed"
}
