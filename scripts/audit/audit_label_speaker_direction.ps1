[CmdletBinding()]
param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Config = "configs\eptnet_marlin11_aligned_windowed_seed42.yaml",
    [string]$JsonOutput = "results\audits\marlin11_label_speaker_direction.json",
    [string]$MarkdownOutput = "results\audits\marlin11_label_speaker_direction.md"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot
$env:PYTHONPATH = (Resolve-Path ".\src").Path

& $Python scripts\audit\audit_label_speaker_direction.py `
    --config $Config `
    --json-output $JsonOutput `
    --markdown-output $MarkdownOutput
if ($LASTEXITCODE -ne 0) {
    throw "Label/speaker direction audit failed"
}
