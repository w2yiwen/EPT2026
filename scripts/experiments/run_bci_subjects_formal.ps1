[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$Device = "cuda:0",
    [int[]]$Seeds = @(13, 42, 73)
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

$matrix = @(
    @{ Config = "configs/bci_subjects.yaml"; Experiment = "eptnet_bci_subjects_no_text" },
    @{ Config = "configs/bci_subjects_baseline_early_fusion_gru.yaml"; Experiment = "baseline_early_fusion_gru_bci_subjects_no_text" },
    @{ Config = "configs/bci_subjects_baseline_fusion_transformer.yaml"; Experiment = "baseline_fusion_transformer_bci_subjects_no_text" },
    @{ Config = "configs/bci_subjects_ablation_no_recurrent_fusion.yaml"; Experiment = "ablation_no_recurrent_fusion_bci_subjects_no_text" }
)

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

Invoke-CheckedPython -Arguments @(
    "-u", "scripts/audit/audit_bci_subjects.py",
    "--config", "configs/bci_subjects.yaml",
    "--json-output", "results/bci_subjects_data_audit.json",
    "--markdown-output", "results/bci_subjects_data_audit.md"
)
Invoke-CheckedPython -Arguments @(
    "-u", "scripts/audit/audit_parameter_fairness.py",
    "configs/bci_subjects.yaml",
    "configs/bci_subjects_baseline_early_fusion_gru.yaml",
    "configs/bci_subjects_baseline_fusion_transformer.yaml",
    "configs/bci_subjects_ablation_no_recurrent_fusion.yaml",
    "--device", "cpu",
    "--output", "results/parameter_fairness_bci_subjects.json"
)

foreach ($spec in $matrix) {
    foreach ($seed in $Seeds) {
        $runDirectory = Join-Path "results" (Join-Path $spec.Experiment "seed_$seed")
        if (Test-Path -LiteralPath $runDirectory) {
            throw "Refusing to reuse an existing formal run directory: $runDirectory"
        }
        Write-Host "TRAIN config=$($spec.Config) seed=$seed device=$Device"
        Invoke-CheckedPython -Arguments @(
            "-u", "-m", "eptnet.train",
            "--config", $spec.Config,
            "--seed", $seed.ToString(),
            "--device", $Device
        )
        $checkpoint = Join-Path $runDirectory "best.pt"
        $metrics = Join-Path $runDirectory "test_metrics.json"
        if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
            throw "Training did not produce the expected checkpoint: $checkpoint"
        }
        Write-Host "EVALUATE experiment=$($spec.Experiment) seed=$seed"
        Invoke-CheckedPython -Arguments @(
            "-u", "-m", "eptnet.evaluate",
            "--config", $spec.Config,
            "--checkpoint", $checkpoint,
            "--output", $metrics,
            "--device", $Device
        )
        if (-not (Test-Path -LiteralPath $metrics -PathType Leaf)) {
            throw "Evaluation did not produce the expected metrics: $metrics"
        }
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

Write-Host "BCI-subject formal experiment matrix completed successfully."
