[CmdletBinding()]
param(
    [string]$Source = "data\raw\bci_subjects_ept_v1",
    [switch]$Overwrite
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "prepare_whisper_alignment12_4060.ps1") `
    -Source $Source `
    -ExcludeSession @("session_011") `
    -Overwrite:$Overwrite
exit $LASTEXITCODE
