[CmdletBinding()]
param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Config = "configs/default.yaml",
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [string]$Output = "",
    [string]$Device = "cuda:0",
    [switch]$NoCalibration
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot
$env:PYTHONPATH = (Resolve-Path ".\src").Path

if (Test-Path -LiteralPath $Python) {
    $pythonCommand = (Resolve-Path -LiteralPath $Python).Path
}
else {
    $pythonLookup = Get-Command $Python -ErrorAction SilentlyContinue
    if ($null -eq $pythonLookup) {
        throw "Python interpreter was not found: $Python"
    }
    $pythonCommand = $pythonLookup.Source
}

New-Item -ItemType Directory -Force ".\logs" | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = ".\logs\evaluate_$stamp.log"
$arguments = @(
    "-u", "-m", "eptnet.evaluate",
    "--config", $Config,
    "--checkpoint", $Checkpoint,
    "--device", $Device
)
if ($Output) {
    $arguments += @("--output", $Output)
}
if ($NoCalibration) {
    $arguments += "--no-calibration"
}

Write-Host "Starting foreground evaluation; log=$logPath"
& $pythonCommand @arguments 2>&1 | Tee-Object -FilePath $logPath
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Error "Evaluation failed with exit code $exitCode. See $logPath"
    exit $exitCode
}

Write-Host "Evaluation completed successfully; log=$logPath"
exit 0
