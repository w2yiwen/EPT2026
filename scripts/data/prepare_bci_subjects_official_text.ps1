param(
    [Parameter(Mandatory = $false)]
    [string]$Source = "..\BCI",
    [Parameter(Mandatory = $false)]
    [string]$Output = "data\processed\bci_subjects_ept_v2_macbert",
    [Parameter(Mandatory = $false)]
    [string]$RawOutput = "data\raw\bci_subjects_ept_v2_macbert",
    [Parameter(Mandatory = $false)]
    [string]$ModelCache = "..\third_party\research_candidates\hf_cache"
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")
$env:PYTHONPATH = (Resolve-Path ".\src").Path
$env:USE_TF = "0"
$env:TRANSFORMERS_NO_TF = "1"
python -m eptnet.data.prepare_bci_subjects `
    --source $Source `
    --output $Output `
    --raw-output $RawOutput `
    --dataset-name bci_subjects_ept_v2_macbert `
    --video-backend legacy `
    --audio-backend none `
    --text-backend macbert `
    --behavior-device cuda `
    --behavior-batch-size 32 `
    --model-cache $ModelCache `
    --local-files-only `
    --window-size 64 `
    --stride 32 `
    --min-window-size 16
