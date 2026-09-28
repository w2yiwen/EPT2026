[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$Source = "data\raw\bci_subjects_ept_v1",
    [Parameter(Mandatory = $false)]
    [string]$Output = "data\processed\bci_subjects_ept_v6_marlin_complete12",
    [Parameter(Mandatory = $false)]
    [string]$DatasetName = "bci_subjects_ept_v6_marlin_complete12",
    [Parameter(Mandatory = $false)]
    [string]$MarlinCache = "data\cache\bci_subjects_ept_v6_marlin_complete12\marlin",
    [Parameter(Mandatory = $false)]
    [string]$WhisperAlignment = "data\cache\bci_subjects_ept_v6_marlin_complete12\whisper_alignment",
    [Parameter(Mandatory = $false)]
    [string[]]$ExcludeSession = @(),
    [Parameter(Mandatory = $false)]
    [string]$MarlinCheckpoint = "src\eptnet\models\behavior\face\marlin\assets\marlin_vit_small_ytf.encoder.pt",
    [Parameter(Mandatory = $false)]
    [string]$BehaviorDevice = "cuda:0",
    [Parameter(Mandatory = $false)]
    [int]$BehaviorBatchSize = 4,
    [Parameter(Mandatory = $false)]
    [switch]$Overwrite
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")
$env:PYTHONPATH = (Resolve-Path ".\src").Path

if (-not (Test-Path -LiteralPath $MarlinCheckpoint -PathType Leaf)) {
    throw "MARLIN checkpoint missing: $MarlinCheckpoint. Run scripts\data\install_behavior_encoder_assets.py first."
}
$expectedAlignmentCount = 12 - $ExcludeSession.Count
if (@(Get-ChildItem -LiteralPath $WhisperAlignment -Filter "session_*.json" -File -ErrorAction SilentlyContinue).Count -lt $expectedAlignmentCount) {
    throw "Expected at least $expectedAlignmentCount Whisper alignment artifacts in $WhisperAlignment. Run prepare_whisper_alignment12.ps1 first."
}

$gpuCheck = & .\.venv\Scripts\python.exe -c "import torch,sys; ok=torch.cuda.is_available(); print(torch.cuda.get_device_name(0) if ok else 'CUDA unavailable'); sys.exit(0 if ok else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "CUDA is unavailable in .venv; MARLIN preparation requires the NVIDIA GPU path."
}
Write-Host "MARLIN device: $gpuCheck"

$arguments = @(
    "-m", "eptnet.data.prepare_bci_subjects",
    "--source", $Source,
    "--raw-output", $Source,
    "--output", $Output,
    "--dataset-name", $DatasetName,
    "--reuse-staged-raw",
    "--required-profile", "strong_behavior_features",
    "--video-backend", "marlin",
    "--audio-backend", "wavlm",
    "--text-backend", "macbert",
    "--behavior-device", $BehaviorDevice,
    "--behavior-batch-size", $BehaviorBatchSize,
    "--marlin-checkpoint", $MarlinCheckpoint,
    "--marlin-cache-dir", $MarlinCache,
    "--whisper-alignment-dir", $WhisperAlignment,
    "--require-whisper-alignment",
    "--marlin-crop-face",
    "--local-files-only",
    "--complete-behavior-sources-only",
    "--alignment-mode", "session_registered_time",
    "--no-include-facial-csv",
    "--no-write-compatibility-windows",
    "--window-size", "64",
    "--stride", "32",
    "--min-window-size", "16"
)
foreach ($sessionId in $ExcludeSession) {
    $arguments += @("--exclude-session", $sessionId)
}
if ($Overwrite) {
    $arguments += "--overwrite"
}

& .\.venv\Scripts\python.exe @arguments
exit $LASTEXITCODE
