param(
    [Parameter(Mandatory = $false)]
    [string]$Source = "data\raw\bci_subjects_ept_v1",
    [Parameter(Mandatory = $false)]
    [string]$Output = "data\processed\bci_subjects_ept_v6_behavior_complete12",
    [Parameter(Mandatory = $false)]
    [string]$DatasetName = "bci_subjects_ept_v6_behavior_complete12",
    [Parameter(Mandatory = $false)]
    [string]$OpenFaceCache = "data\cache\bci_subjects_ept_v6_behavior_complete12\openface",
    [Parameter(Mandatory = $false)]
    [string]$BehaviorDevice = "cuda:0",
    [Parameter(Mandatory = $false)]
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")
$env:PYTHONPATH = (Resolve-Path ".\src").Path

$arguments = @(
    "-m", "eptnet.data.prepare_bci_subjects",
    "--source", $Source,
    "--raw-output", $Source,
    "--output", $Output,
    "--dataset-name", $DatasetName,
    "--reuse-staged-raw",
    "--required-profile", "strong_behavior_features",
    "--video-backend", "openface",
    "--audio-backend", "wavlm",
    "--text-backend", "macbert",
    "--behavior-device", $BehaviorDevice,
    "--openface-cache-dir", $OpenFaceCache,
    "--local-files-only",
    "--complete-behavior-sources-only",
    "--alignment-mode", "session_registered_time",
    "--no-include-facial-csv",
    "--no-write-compatibility-windows",
    "--window-size", "64",
    "--stride", "32",
    "--min-window-size", "16"
)
if ($Overwrite) {
    $arguments += "--overwrite"
}

& .\.venv\Scripts\python.exe @arguments
exit $LASTEXITCODE
