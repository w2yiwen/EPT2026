[CmdletBinding()]
param([switch]$Overwrite)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
$env:PYTHONPATH = (Resolve-Path ".\src").Path
$arguments = @("scripts\data\remap_legacy_marlin_cache_aligned11.py")
if ($Overwrite) { $arguments += "--overwrite" }
& .\.venv\Scripts\python.exe @arguments
exit $LASTEXITCODE
