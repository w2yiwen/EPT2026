param(
    [Parameter(Mandatory = $false)]
    [string]$Source = "data\raw\bci_subjects_ept_v1",
    [Parameter(Mandatory = $false)]
    [string]$Output = "data\processed\bci_subjects_ept_v4_no_facial",
    [Parameter(Mandatory = $false)]
    [string]$DatasetName = "bci_subjects_ept_v4_no_facial",
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
    "--required-profile", "model_contract",
    "--video-backend", "legacy",
    "--audio-backend", "none",
    "--text-backend", "hash",
    "--no-include-facial-csv",
    "--no-write-compatibility-windows",
    "--window-size", "64",
    "--stride", "32",
    "--min-window-size", "16"
)
if ($Overwrite) {
    $arguments += "--overwrite"
}

python @arguments
