[CmdletBinding()]
param(
    [string]$Source = "data\raw\bci_subjects_ept_v1",
    [string]$Output = "data\cache\bci_subjects_ept_v6_marlin_complete12\whisper_alignment",
    [string]$Device = "cuda:0",
    [string]$Model = "turbo",
    [double]$MinimumReferenceCoverage = 0.50,
    [string[]]$ExcludeSession = @(),
    [switch]$Overwrite
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
$env:PYTHONPATH = (Resolve-Path ".\src").Path

$arguments = @(
    "scripts\data\generate_whisper_alignment12.py",
    "--source", $Source,
    "--output", $Output,
    "--device", $Device,
    "--model", $Model,
    "--minimum-reference-coverage", $MinimumReferenceCoverage
)
foreach ($sessionId in $ExcludeSession) {
    $arguments += @("--exclude-session", $sessionId)
}
if ($Overwrite) { $arguments += "--overwrite" }
& .\.venv\Scripts\python.exe @arguments
exit $LASTEXITCODE
