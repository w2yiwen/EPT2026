[CmdletBinding()]
param(
    [int]$RefreshSeconds = 5
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$cacheRoot = Join-Path $projectRoot "data\cache\bci_subjects_ept_v6_marlin_complete12\marlin"
$alignmentRoot = Join-Path $projectRoot "data\cache\bci_subjects_ept_v6_marlin_complete12\whisper_alignment"
$datasetRoot = Join-Path $projectRoot "data\processed\bci_subjects_ept_v6_marlin_complete12"
$successPath = Join-Path $datasetRoot "_SUCCESS.json"
$totalSessions = 12
$lastProgress = ""

Write-Host "Watching MARLIN12 preprocessing. Press Ctrl+C to stop watching only."
while ($true) {
    $completed = @(
        Get-ChildItem -LiteralPath $cacheRoot -Filter "*.npz" -File -ErrorAction SilentlyContinue
    ).Count
    $aligned = @(Get-ChildItem -LiteralPath $alignmentRoot -Filter "session_*.json" -File -ErrorAction SilentlyContinue).Count
    $percent = [math]::Min(100, [math]::Floor(50 * ($aligned + $completed) / $totalSessions))
    $pythonPreparation = @(
        Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
            Where-Object {
                $_.CommandLine -like "*eptnet.data.prepare_bci_subjects*" -and
                $_.CommandLine -like "*bci_subjects_ept_v6_marlin_complete12*"
            }
    )
    $pythonAlignment = @(
        Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -like "*generate_whisper_alignment12.py*" }
    )

    $stage = if ($pythonAlignment.Count -gt 0) {
        "Whisper text plus video-audio alignment"
    }
    elseif ($pythonPreparation.Count -gt 0) {
        "MARLIN/WavLM/MacBERT CUDA encoding or dataset writing"
    }
    elseif (Test-Path -LiteralPath $successPath) {
        "completed"
    }
    else {
        "not running"
    }

    Write-Progress `
        -Activity "MARLIN12 all-modal preprocessing" `
        -Status "$aligned/$totalSessions aligned; $completed/$totalSessions MARLIN cached; $stage" `
        -PercentComplete $percent

    $progressKey = "$aligned/$completed/$stage"
    if ($progressKey -ne $lastProgress) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] alignment: $aligned/$totalSessions; MARLIN: $completed/$totalSessions; $stage"
        $lastProgress = $progressKey
    }

    if (Test-Path -LiteralPath $successPath) {
        Write-Progress -Activity "MARLIN12 all-modal preprocessing" -Completed
        Write-Host "Completed: $successPath"
        exit 0
    }
    if ($pythonPreparation.Count -eq 0 -and $pythonAlignment.Count -eq 0) {
        Write-Progress -Activity "MARLIN12 all-modal preprocessing" -Completed
        Write-Warning "Preprocessing is not running and _SUCCESS.json is absent. Restart the preparation command."
        exit 1
    }
    Start-Sleep -Seconds $RefreshSeconds
}
