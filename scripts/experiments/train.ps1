[CmdletBinding()]
param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Config = "configs/default.yaml",
    [Nullable[int]]$Seed = $null,
    [string]$OutputDir = "",
    [string]$Device = "cuda:0",
    [string]$Resume = "",
    [switch]$Smoke,
    [switch]$NoProgress
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
$logPath = ".\logs\train_$stamp.log"
$arguments = @("-u", "-m", "eptnet.train", "--config", $Config, "--device", $Device)
if ($NoProgress) {
    $arguments += "--no-progress"
}
else {
    $arguments += "--progress"
}
if ($null -ne $Seed) {
    $arguments += @("--seed", $Seed.ToString())
}
if ($OutputDir) {
    $arguments += @("--output-dir", $OutputDir)
}
if ($Resume) {
    $arguments += @("--resume", $Resume)
}
if ($Smoke) {
    $arguments += "--smoke"
}

Write-Host "Starting foreground training; log=$logPath"
$transcriptStarted = $false
try {
    Start-Transcript -LiteralPath $logPath -Force | Out-Null
    $transcriptStarted = $true
    & $pythonCommand @arguments
    $exitCode = $LASTEXITCODE
}
finally {
    if ($transcriptStarted) {
        Stop-Transcript | Out-Null
    }
}
if ($exitCode -ne 0) {
    Write-Error "Training failed with exit code $exitCode. See $logPath"
    exit $exitCode
}

Write-Host "Training completed successfully; log=$logPath"
exit 0
