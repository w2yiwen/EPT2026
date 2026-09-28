[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$Device = "cuda:0"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$codeRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $codeRoot
$env:PYTHONPATH = (Resolve-Path ".\src").Path

$primary = @(
    @{ Config = "configs/default.yaml"; Experiment = "eptnet_continuous_no_text" },
    @{ Config = "configs/baseline_early_fusion_gru.yaml"; Experiment = "baseline_early_fusion_gru_continuous_no_text" },
    @{ Config = "configs/baseline_fusion_transformer.yaml"; Experiment = "baseline_fusion_transformer_continuous_no_text" },
    @{ Config = "configs/ablation_no_persistent.yaml"; Experiment = "ablation_no_persistent_continuous_no_text" }
)
$diagnostics = @(
    @{ Config = "configs/ablation_fixed_reader.yaml"; Experiment = "ablation_fixed_reader_continuous_no_text" },
    @{ Config = "configs/ablation_no_behavior.yaml"; Experiment = "ablation_no_behavior_continuous_no_text" },
    @{ Config = "configs/ablation_single_scale.yaml"; Experiment = "ablation_single_scale_continuous_no_text" }
)
$seeds = @(13, 42, 73)

function Invoke-Python {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Invoke-Run {
    param([hashtable]$Spec, [int]$Seed)
    $run = Join-Path "results" (Join-Path $Spec.Experiment "seed_$Seed")
    $metrics = Join-Path $run "test_metrics.json"
    if (Test-Path -LiteralPath $metrics) {
        Write-Host "SKIP completed experiment=$($Spec.Experiment) seed=$Seed"
        return
    }
    if (Test-Path -LiteralPath $run) {
        throw "Incomplete run directory requires manual inspection: $run"
    }
    Write-Host "TRAIN experiment=$($Spec.Experiment) seed=$Seed"
    Invoke-Python -Arguments @(
        "-u", "-m", "eptnet.train", "--config", $Spec.Config,
        "--seed", $Seed.ToString(), "--device", $Device
    )
    $checkpoint = Join-Path $run "best.pt"
    Write-Host "EVALUATE experiment=$($Spec.Experiment) seed=$Seed"
    Invoke-Python -Arguments @(
        "-u", "-m", "eptnet.evaluate", "--config", $Spec.Config,
        "--checkpoint", $checkpoint, "--output", $metrics, "--device", $Device
    )
}

foreach ($spec in $primary) {
    foreach ($seed in $seeds) {
        Invoke-Run -Spec $spec -Seed $seed
    }
    $directory = Join-Path "results" $spec.Experiment
    $inputs = @(
        foreach ($seed in $seeds) {
            Join-Path $directory (Join-Path "seed_$seed" "test_metrics.json")
        }
    )
    $aggregateArguments = @("-u", "-m", "eptnet.aggregate") + $inputs + @(
        "--output", (Join-Path $directory "aggregate.json"),
        "--csv", (Join-Path $directory "aggregate.csv")
    )
    Invoke-Python -Arguments $aggregateArguments
}

foreach ($spec in $diagnostics) {
    Invoke-Run -Spec $spec -Seed 42
}

Write-Host "REMAINING FORMAL MATRIX COMPLETE"
