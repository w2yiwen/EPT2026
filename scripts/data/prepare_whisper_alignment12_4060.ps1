[CmdletBinding()]
param(
    [string]$Source = "data\raw\bci_subjects_ept_v1",
    [double]$MinimumReferenceCoverage = 0.50,
    [string[]]$ExcludeSession = @("session_011"),
    [switch]$Overwrite
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
$gpuName = & .\.venv\Scripts\python.exe -c "import torch,sys; ok=torch.cuda.is_available(); print(torch.cuda.get_device_name(0) if ok else 'CUDA unavailable'); sys.exit(0 if ok else 1)"
if ($LASTEXITCODE -ne 0 -or $gpuName -notmatch "4060") {
    throw "The isolated 4060 entry point requires an RTX 4060 on cuda:0; detected '$gpuName'."
}
$arguments = @{
    Source = $Source
    Output = "data\cache\bci_subjects_ept_v6_marlin4060_complete12\whisper_alignment"
    Device = "cuda:0"
    Model = "turbo"
    MinimumReferenceCoverage = $MinimumReferenceCoverage
    ExcludeSession = $ExcludeSession
    Overwrite = $Overwrite
}
& (Join-Path $PSScriptRoot "prepare_whisper_alignment12.ps1") @arguments
exit $LASTEXITCODE
