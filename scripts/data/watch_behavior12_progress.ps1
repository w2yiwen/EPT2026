[CmdletBinding()]
param(
    [int]$RefreshSeconds = 5
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$cacheRoot = Join-Path $projectRoot "data\cache\bci_subjects_ept_v6_behavior_complete12\openface"
$datasetRoot = Join-Path $projectRoot "data\processed\bci_subjects_ept_v6_behavior_complete12"
$successPath = Join-Path $datasetRoot "_SUCCESS.json"
$totalSessions = 12
$lastCompleted = -1

Write-Host "Watching behavior12 preprocessing. Press Ctrl+C to stop watching only."
while ($true) {
    $completed = @(
        Get-ChildItem -LiteralPath $cacheRoot -Filter "*.csv" -File -ErrorAction SilentlyContinue
    ).Count
    $percent = [math]::Min(100, [math]::Floor(100 * $completed / $totalSessions))
    $openFace = Get-Process -Name "FeatureExtraction" -ErrorAction SilentlyContinue |
        Sort-Object StartTime |
        Select-Object -Last 1
    $pythonPreparation = @(
        Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -like "*eptnet.data.prepare_bci_subjects*" }
    )

    $stage = if ($null -ne $openFace) {
        $elapsed = (Get-Date) - $openFace.StartTime
        "OpenFace active; elapsed=$($elapsed.ToString('hh\:mm\:ss'))"
    }
    elseif ($pythonPreparation.Count -gt 0) {
        "CUDA encoding / validation / dataset writing"
    }
    elseif (Test-Path -LiteralPath $successPath) {
        "completed"
    }
    else {
        "not running"
    }

    Write-Progress `
        -Activity "behavior12 all-modal preprocessing" `
        -Status "$completed/$totalSessions OpenFace videos cached; $stage" `
        -PercentComplete $percent

    if ($completed -ne $lastCompleted) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] OpenFace cache: $completed/$totalSessions; $stage"
        $lastCompleted = $completed
    }

    if (Test-Path -LiteralPath $successPath) {
        Write-Progress -Activity "behavior12 all-modal preprocessing" -Completed
        $summaryPath = Join-Path $datasetRoot "dataset_summary.json"
        if (Test-Path -LiteralPath $summaryPath) {
            $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
            Write-Host "Completed: $successPath"
            Write-Host "Sessions: $($summary.num_subjects); split=$($summary.sessions_per_split | ConvertTo-Json -Compress)"
        }
        exit 0
    }

    if ($null -eq $openFace -and $pythonPreparation.Count -eq 0) {
        Write-Progress -Activity "behavior12 all-modal preprocessing" -Completed
        Write-Warning "Preprocessing is not running and _SUCCESS.json is absent. Restart the preparation command."
        exit 1
    }

    Start-Sleep -Seconds $RefreshSeconds
}
