[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$Source = "data\raw\bci_subjects_ept_v1",
    [string[]]$ExcludeSession = @("session_011"),
    [Parameter(Mandatory = $false)]
    [switch]$Overwrite
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot

$gpuName = & .\.venv\Scripts\python.exe -c "import torch,sys; ok=torch.cuda.is_available(); print(torch.cuda.get_device_name(0) if ok else 'CUDA unavailable'); sys.exit(0 if ok else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "CUDA is unavailable in .venv."
}
if ($gpuName -notmatch "4060") {
    throw "The isolated 4060 entry point requires an RTX 4060, but cuda:0 is '$gpuName'. Use prepare_current_session_registered_marlin12.ps1 for the 4090 route."
}
Write-Host "Isolated RTX 4060 route: $gpuName; behavior batch size=4"

$arguments = @{
    Source = $Source
    Output = "data\processed\bci_subjects_ept_v6_marlin4060_aligned11"
    DatasetName = "bci_subjects_ept_v6_marlin4060_aligned11"
    MarlinCache = "data\cache\bci_subjects_ept_v6_marlin4060_aligned11\marlin"
    # Keep the already completed candidate-cohort alignments; session_011 is
    # excluded explicitly and is never required or consumed.
    WhisperAlignment = "data\cache\bci_subjects_ept_v6_marlin4060_complete12\whisper_alignment"
    BehaviorDevice = "cuda:0"
    BehaviorBatchSize = 4
    ExcludeSession = $ExcludeSession
    Overwrite = $Overwrite
}

& (Join-Path $PSScriptRoot "prepare_current_session_registered_marlin12.ps1") @arguments
exit $LASTEXITCODE
