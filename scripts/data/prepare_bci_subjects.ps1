param(
    [Parameter(Mandatory = $false)]
    [string]$Source = "..\BCI",
    [Parameter(Mandatory = $false)]
    [string]$Output = "data\processed\bci_subjects_ept_v1",
    [Parameter(Mandatory = $false)]
    [string]$RawOutput = "data\raw\bci_subjects_ept_v1",
    [Parameter(Mandatory = $false)]
    [ValidateSet(
        "model_contract",
        "legacy_face_text",
        "strong_behavior_sources",
        "strong_behavior_features",
        "complete_multimodal_sources"
    )]
    [string]$RequiredProfile = "model_contract",
    [Parameter(Mandatory = $false)]
    [ValidateSet("legacy", "openface")]
    [string]$VideoBackend = "legacy",
    [Parameter(Mandatory = $false)]
    [ValidateSet("none", "wavlm")]
    [string]$AudioBackend = "none",
    [Parameter(Mandatory = $false)]
    [ValidateSet("hash", "macbert")]
    [string]$TextBackend = "hash",
    [Parameter(Mandatory = $false)]
    [string]$BehaviorDevice = "auto"
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")
$env:PYTHONPATH = (Resolve-Path ".\src").Path
python -m eptnet.data.prepare_bci_subjects `
    --source $Source `
    --output $Output `
    --raw-output $RawOutput `
    --required-profile $RequiredProfile `
    --video-backend $VideoBackend `
    --audio-backend $AudioBackend `
    --text-backend $TextBackend `
    --behavior-device $BehaviorDevice `
    --window-size 64 `
    --stride 32 `
    --min-window-size 16
