[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$Device = "cuda:0",
    [int[]]$Seeds = @(13, 42, 73),
    [switch]$SkipDiagnostics
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot
$env:PYTHONPATH = (Resolve-Path ".\src").Path

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "Python executable was not found: $Python"
}
if ($Seeds.Count -lt 2) {
    throw "Formal comparisons require at least two distinct seeds"
}
if (($Seeds | Sort-Object -Unique).Count -ne $Seeds.Count) {
    throw "Seeds must be unique"
}

$primaryMatrix = @(
    @{ Config = "configs/default.yaml"; Experiment = "eptnet_continuous_no_text" },
    @{ Config = "configs/baseline_early_fusion_gru.yaml"; Experiment = "baseline_early_fusion_gru_continuous_no_text" },
    @{ Config = "configs/baseline_fusion_transformer.yaml"; Experiment = "baseline_fusion_transformer_continuous_no_text" },
    @{ Config = "configs/ablation_no_persistent.yaml"; Experiment = "ablation_no_persistent_continuous_no_text" }
)
$diagnosticMatrix = @(
    @{ Config = "configs/ablation_fixed_reader.yaml"; Experiment = "ablation_fixed_reader_continuous_no_text" },
    @{ Config = "configs/ablation_no_behavior.yaml"; Experiment = "ablation_no_behavior_continuous_no_text" },
    @{ Config = "configs/ablation_single_scale.yaml"; Experiment = "ablation_single_scale_continuous_no_text" }
)

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Invoke-Run {
    param(
        [hashtable]$Spec,
        [int]$Seed
    )
    $runDirectory = Join-Path "results" (Join-Path $Spec.Experiment "seed_$Seed")
    if (Test-Path $runDirectory) {
        throw "Refusing to reuse an existing formal run directory: $runDirectory"
    }
    Write-Host "TRAIN config=$($Spec.Config) seed=$Seed device=$Device"
    Invoke-CheckedPython -Arguments @(
        "-u", "-m", "eptnet.train",
        "--config", $Spec.Config,
        "--seed", $Seed.ToString(),
        "--device", $Device
    )
    $checkpoint = Join-Path $runDirectory "best.pt"
    $metrics = Join-Path $runDirectory "test_metrics.json"
    if (-not (Test-Path $checkpoint)) {
        throw "Training did not produce the expected checkpoint: $checkpoint"
    }
    Write-Host "EVALUATE experiment=$($Spec.Experiment) seed=$Seed"
    Invoke-CheckedPython -Arguments @(
        "-u", "-m", "eptnet.evaluate",
        "--config", $Spec.Config,
        "--checkpoint", $checkpoint,
        "--output", $metrics,
        "--device", $Device
    )
    if (-not (Test-Path $metrics)) {
        throw "Evaluation did not produce the expected metrics: $metrics"
    }
}

foreach ($spec in $primaryMatrix) {
    foreach ($seed in $Seeds) {
        Invoke-Run -Spec $spec -Seed $seed
    }
    $experimentDirectory = Join-Path "results" $spec.Experiment
    $metricFiles = @(
        foreach ($seed in $Seeds) {
            Join-Path $experimentDirectory (Join-Path "seed_$seed" "test_metrics.json")
        }
    )
    $aggregateArguments = @(
        "-u", "-m", "eptnet.aggregate"
    ) + $metricFiles + @(
        "--output", (Join-Path $experimentDirectory "aggregate.json"),
        "--csv", (Join-Path $experimentDirectory "aggregate.csv")
    )
    Invoke-CheckedPython -Arguments $aggregateArguments
}

if (-not $SkipDiagnostics) {
    foreach ($spec in $diagnosticMatrix) {
        Invoke-Run -Spec $spec -Seed 42
    }
}

Write-Host "Formal experiment matrix completed successfully."
