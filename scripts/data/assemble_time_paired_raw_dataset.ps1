param(
    [string]$TargetRoot = "",
    [Parameter(Mandatory = $true)][string]$OpenBciRoot,
    [Parameter(Mandatory = $true)][string]$PpgDeviceRoot
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
if (-not $TargetRoot) {
    $TargetRoot = Join-Path $projectRoot "code\data\raw\bci_subjects_ept_v1"
}
$target = (Resolve-Path $TargetRoot).Path
$expectedTarget = [IO.Path]::GetFullPath((Join-Path $projectRoot "code\data\raw\bci_subjects_ept_v1"))
if ($target -ne $expectedTarget) {
    throw "Refusing unexpected target: $target"
}

$exportRoot = (Resolve-Path (Join-Path $projectRoot "BCI\collection\documents-export-2026-9-27")).Path
$collectionRoot = (Resolve-Path (Join-Path $projectRoot "BCI\sensor_text_collection_2025-03-05_to_03-08")).Path
$openBciRoot = (Resolve-Path -LiteralPath $OpenBciRoot).Path
$ppgDeviceRoot = (Resolve-Path -LiteralPath $PpgDeviceRoot).Path

function Get-Hash([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

function Copy-Verified([string]$Source, [string]$Destination, [switch]$Replace) {
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Missing source: $Source"
    }
    $parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $sourceHash = Get-Hash $Source
    if (Test-Path -LiteralPath $Destination) {
        $destinationHash = Get-Hash $Destination
        if ($sourceHash -eq $destinationHash) { return }
        if (-not $Replace) {
            throw "Destination differs and replacement was not authorized by this operation: $Destination"
        }
    }
    Copy-Item -LiteralPath $Source -Destination $Destination -Force:$Replace
    if ((Get-Hash $Destination) -ne $sourceHash) {
        throw "Checksum mismatch after copy: $Destination"
    }
}

function New-FileRecord(
    [string]$SessionId,
    [string]$SubjectId,
    [string]$Source,
    [string]$Destination,
    [string]$Confidence,
    [string]$Basis
) {
    $item = Get-Item -LiteralPath $Source
    return [pscustomobject][ordered]@{
        session_id = $SessionId
        subject_id = $SubjectId
        source_relative_path = [IO.Path]::GetRelativePath($projectRoot, $item.FullName).Replace("\", "/")
        source_absolute_path = $item.FullName
        staged_relative_path = [IO.Path]::GetRelativePath($target, $Destination).Replace("\", "/")
        bytes = [int64]$item.Length
        sha256 = Get-Hash $Source
        pairing_confidence = $Confidence
        pairing_basis = $Basis
    }
}

function Write-Json([string]$Path, $Value) {
    $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Set-SessionMetadata(
    [string]$SessionId,
    [string]$SubjectId,
    [object[]]$NewRecords,
    [hashtable]$Available,
    [object]$Pairing
) {
    $sessionRoot = Join-Path $target $SessionId
    $metadataPath = Join-Path $sessionRoot "session_metadata.json"
    $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    $newPaths = @($NewRecords | ForEach-Object { $_.staged_relative_path })
    $metadata.files = @($metadata.files | Where-Object { $newPaths -notcontains $_.staged_relative_path }) + @($NewRecords)
    foreach ($key in $Available.Keys) {
        if ($metadata.available_sources.PSObject.Properties.Name -contains $key) {
            $metadata.available_sources.$key = [bool]$Available[$key]
        } else {
            $metadata.available_sources | Add-Member -NotePropertyName $key -NotePropertyValue ([bool]$Available[$key])
        }
        if ($Available[$key] -and $metadata.missing_source_reasons) {
            $metadata.missing_source_reasons.PSObject.Properties.Remove($key)
        }
    }
    $metadata | Add-Member -NotePropertyName time_pairing -NotePropertyValue $Pairing -Force
    Write-Json $metadataPath $metadata
    Write-Json (Join-Path $sessionRoot "source_manifest.json") $metadata
}

$rootManifestPath = Join-Path $target "source_manifest.json"
$rootManifest = Get-Content -LiteralPath $rootManifestPath -Raw | ConvertFrom-Json
$sessionIndexPath = Join-Path $target "session_index.json"
$sessionIndex = Get-Content -LiteralPath $sessionIndexPath -Raw | ConvertFrom-Json
$audit = [System.Collections.Generic.List[object]]::new()

function Upsert-RootFiles([object[]]$Records) {
    $paths = @($Records | ForEach-Object { $_.staged_relative_path })
    $script:rootManifest.files = @($script:rootManifest.files | Where-Object { $paths -notcontains $_.staged_relative_path }) + @($Records)
}

function Add-Audit(
    [string]$SessionId,
    [string]$SubjectId,
    [string]$Modality,
    [string]$Status,
    [string]$Confidence,
    [string]$Source,
    [string]$Destination,
    [string]$Evidence
) {
    $script:audit.Add([pscustomobject][ordered]@{
        session_id = $SessionId
        subject_id = $SubjectId
        modality = $Modality
        status = $Status
        confidence = $Confidence
        source = $Source
        destination = $Destination
        evidence = $Evidence
    })
}

# Repair HSY's primary PPG using the pair that starts 47 seconds after its EEG.
$hsyRoot = Join-Path $target "session_004"
$hsyOld1 = Join-Path $hsyRoot "session_004_ppg_channel_1.txt"
$hsyOld2 = Join-Path $hsyRoot "session_004_ppg_channel_2.txt"
$hsyBackup1 = Join-Path $hsyRoot "session_004_ppg_channel_1_earlier_short_candidate_205609.txt"
$hsyBackup2 = Join-Path $hsyRoot "session_004_ppg_channel_2_earlier_short_candidate_205609.txt"
if (-not (Test-Path -LiteralPath $hsyBackup1)) { Copy-Verified $hsyOld1 $hsyBackup1 }
if (-not (Test-Path -LiteralPath $hsyBackup2)) { Copy-Verified $hsyOld2 $hsyBackup2 }
$hsyNew1 = Join-Path $ppgDeviceRoot "03-05-21-03-04_ch2.txt"
$hsyNew2 = Join-Path $ppgDeviceRoot "03-05-21-03-04_ch3.txt"
Copy-Verified $hsyNew1 $hsyOld1 -Replace
Copy-Verified $hsyNew2 $hsyOld2 -Replace
$hsyBasis = "EEG starts 2025-03-05 21:02:17 and this PPG starts 21:03:04 (47 s later); both replacement channels have 485,470 rows, while the preserved earlier pair has only 3,057 rows/channel."
$hsyRecords = @(
    (New-FileRecord "session_004" "hsy" $hsyNew1 $hsyOld1 "high" $hsyBasis),
    (New-FileRecord "session_004" "hsy" $hsyNew2 $hsyOld2 "high" $hsyBasis),
    (New-FileRecord "session_004" "hsy" (Join-Path $projectRoot "BCI\hsy\hsy03-05-20-56-09_ch2 (1).txt") $hsyBackup1 "low" "Preserved earlier short acquisition; not preferred for reprocessing."),
    (New-FileRecord "session_004" "hsy" (Join-Path $projectRoot "BCI\hsy\hsy03-05-20-56-09_ch2 (2).txt") $hsyBackup2 "low" "Preserved earlier short acquisition; not preferred for reprocessing.")
)
Set-SessionMetadata "session_004" "hsy" $hsyRecords @{ paired_ppg = $true } ([pscustomobject]@{
    status = "time_matched_primary_replaced_with_backup"
    confidence = "high"
    evidence = $hsyBasis
    preferred_ppg = @("session_004_ppg_channel_1.txt", "session_004_ppg_channel_2.txt")
})
Upsert-RootFiles $hsyRecords
Add-Audit "session_004" "hsy" "PPG" "assigned_primary" "high" "$hsyNew1 | $hsyNew2" "$hsyOld1 | $hsyOld2" $hsyBasis

# Direct-name EEG recovery for ZHJ and ZYL.
$directEeg = @(
    [pscustomobject]@{ Session = "session_015"; Subject = "zhj"; File = "OpenBCI-RAW-2025-03-06_19-07-11-ZhangHeJia.txt"; Evidence = "Filename explicitly names ZhangHeJia; EEG starts 19:07:11 and named PPG starts 19:13:43." },
    [pscustomobject]@{ Session = "session_017"; Subject = "zyl"; File = "OpenBCI-RAW-2025-03-05_21-50-13_ZhaoYunLu.txt"; Evidence = "Filename explicitly names ZhaoYunLu; named PPG starts 21:49:22 and EEG starts 21:50:13." }
)
foreach ($item in $directEeg) {
    $source = Join-Path $exportRoot $item.File
    $sessionRoot = Join-Path $target $item.Session
    $destination = Join-Path $sessionRoot "$($item.Session)_eeg_raw.txt"
    Copy-Verified $source $destination
    $record = New-FileRecord $item.Session $item.Subject $source $destination "high" $item.Evidence
    Set-SessionMetadata $item.Session $item.Subject @($record) @{ eeg = $true } ([pscustomobject]@{
        status = "direct_name_and_time_matched"
        confidence = "high"
        evidence = $item.Evidence
    })
    Upsert-RootFiles @($record)
    Add-Audit $item.Session $item.Subject "EEG" "assigned_primary" "high" $source $destination $item.Evidence
}

function New-SupplementalSession(
    [string]$SessionId,
    [string]$SubjectId,
    [string]$DisplayName,
    [object[]]$FileSpecs,
    [object]$Pairing
) {
    $sessionRoot = Join-Path $target $SessionId
    New-Item -ItemType Directory -Force -Path $sessionRoot | Out-Null
    $records = [System.Collections.Generic.List[object]]::new()
    $available = [ordered]@{
        eeg = $false
        paired_ppg = $false
        facial_actions = $false
        openface_features = $false
        audio = $false
        video = $false
        annotated_transcript = $true
    }
    foreach ($spec in $FileSpecs) {
        $destination = Join-Path $sessionRoot $spec.Destination
        Copy-Verified $spec.Source $destination
        $records.Add((New-FileRecord $SessionId $SubjectId $spec.Source $destination $spec.Confidence $spec.Basis))
        if ($spec.Modality -eq "eeg") { $available.eeg = $true }
        if ($spec.Modality -eq "ppg") { $available.paired_ppg = $true }
        if ($spec.Modality -eq "face") { $available.facial_actions = $true }
    }
    $missing = [ordered]@{}
    foreach ($key in $available.Keys) {
        if (-not $available[$key]) { $missing[$key] = "source_artifact_absent_or_not_reliably_paired" }
    }
    $metadata = [pscustomobject][ordered]@{
        dataset = "bci_subjects_ept_v1"
        session_id = $SessionId
        subject_id = $SubjectId
        subject_display_name = $DisplayName
        acquisition_date = "2025-03-09"
        annotation_mode = "yellow_highlight"
        raw_only_requires_reprocessing = $true
        available_sources = [pscustomobject]$available
        missing_source_reasons = [pscustomobject]$missing
        time_pairing = $Pairing
        files = @($records)
    }
    Write-Json (Join-Path $sessionRoot "session_metadata.json") $metadata
    Write-Json (Join-Path $sessionRoot "source_manifest.json") $metadata
    $recordPaths = @($records | ForEach-Object { $_.staged_relative_path })
    $script:rootManifest.files = @(
        $script:rootManifest.files | Where-Object { $recordPaths -notcontains $_.staged_relative_path }
    ) + @($records)
    if (-not @($script:rootManifest.sessions | Where-Object session_id -eq $SessionId)) {
        $script:rootManifest.sessions = @($script:rootManifest.sessions) + [pscustomobject]@{
            session_id = $SessionId
            subject_id = $SubjectId
            raw_path = "raw/bci_subjects_ept_v1/$SessionId"
            status = "raw_only_requires_reprocessing"
        }
    }
    if (-not @($script:sessionIndex.sessions | Where-Object session_id -eq $SessionId)) {
        $script:sessionIndex.sessions = @($script:sessionIndex.sessions) + [pscustomobject]@{
            session_id = $SessionId
            subject_id = $SubjectId
            split = "unassigned"
            raw_path = "raw/bci_subjects_ept_v1/$SessionId"
            processed_path = $null
            status = "raw_only_requires_reprocessing"
        }
    }
    foreach ($record in $records) {
        Add-Audit $SessionId $SubjectId ($record.staged_relative_path -replace '^.*_', '') "assigned" $record.pairing_confidence $record.source_absolute_path (Join-Path $target $record.staged_relative_path) $record.pairing_basis
    }
}

function Spec([string]$Source, [string]$Destination, [string]$Modality, [string]$Confidence, [string]$Basis) {
    return [pscustomobject]@{ Source = $Source; Destination = $Destination; Modality = $Modality; Confidence = $Confidence; Basis = $Basis }
}

$march9 = @(
    [pscustomobject]@{ Id="session_019"; Subject="xx"; Name="XX"; Docs=@("3-9-xx.docx"); Face="action_info_20250309_101738.csv"; FaceConfidence="medium"; FaceBasis="Face ends 10:26:53; DOCX core creation is 10:28:14 (81 s later)." },
    [pscustomobject]@{ Id="session_020"; Subject="yjy"; Name="YJY"; Docs=@("3-9-yjy.docx"); Face="action_info_20250309_103101.csv"; FaceConfidence="high"; FaceBasis="Face ends 10:36:39; DOCX core creation is 10:37:25 (46 s later)." },
    [pscustomobject]@{ Id="session_021"; Subject="jinpeipei"; Name="金佩佩"; Docs=@("3.9-金佩佩.docx"); Face=$null; FaceConfidence=$null; FaceBasis=$null },
    [pscustomobject]@{ Id="session_022"; Subject="li_unknown"; Name="某位李同学"; Docs=@("3.9-某位李同学.docx"); Face=$null; FaceConfidence=$null; FaceBasis=$null },
    [pscustomobject]@{ Id="session_023"; Subject="ssg"; Name="SSG"; Docs=@("3-9-ssg.docx"); Face=$null; FaceConfidence=$null; FaceBasis=$null },
    [pscustomobject]@{ Id="session_024"; Subject="gww"; Name="GWW"; Docs=@("3-9-gww (1).docx", "3-9-gww.docx"); Face="action_info_20250309_112954.csv"; FaceConfidence="high"; FaceBasis="Face ends 11:35:33; the two GWW DOCX files are created at 11:36:00/11:36:46." },
    [pscustomobject]@{ Id="session_025"; Subject="zhangjunjie"; Name="张俊杰"; Docs=@("3.9-张俊杰.docx"); Face=$null; FaceConfidence=$null; FaceBasis=$null },
    [pscustomobject]@{ Id="session_026"; Subject="lixiangyu"; Name="李翔宇"; Docs=@("3.9-李翔宇.docx"); Face=$null; FaceConfidence=$null; FaceBasis=$null },
    [pscustomobject]@{ Id="session_027"; Subject="liyao"; Name="李尧"; Docs=@("3.9-李尧.docx"); Face=$null; FaceConfidence=$null; FaceBasis=$null },
    [pscustomobject]@{ Id="session_028"; Subject="luoyuhuan"; Name="罗雨欢"; Docs=@("3.9-罗雨欢.docx"); Face=$null; FaceConfidence=$null; FaceBasis=$null },
    [pscustomobject]@{ Id="session_029"; Subject="huangxianzhi"; Name="黄显志"; Docs=@("3.9-黄显志.docx"); Face=$null; FaceConfidence=$null; FaceBasis=$null }
)

foreach ($person in $march9) {
    $specs = [System.Collections.Generic.List[object]]::new()
    for ($index = 0; $index -lt $person.Docs.Count; $index++) {
        $suffix = if ($index -eq 0) { "annotations.docx" } else { "transcript_unmarked_{0:D2}.docx" -f $index }
        $specs.Add((Spec (Join-Path $exportRoot $person.Docs[$index]) "$($person.Id)_$suffix" "annotation" "high" "Participant identity is explicit in the exported document filename."))
    }
    if ($person.Face) {
        $specs.Add((Spec (Join-Path $exportRoot $person.Face) "$($person.Id)_facial_actions.csv" "face" $person.FaceConfidence $person.FaceBasis))
    }
    if ($person.Subject -eq "jinpeipei") {
        $ppgBasis = "Two scalar files explicitly named 金佩佩-1/-2; both contain 98,213 samples and form the only direct-name PPG pair for this participant."
        $specs.Add((Spec (Join-Path $exportRoot "3.9-金佩佩-1.txt") "$($person.Id)_ppg_channel_1.txt" "ppg" "high" $ppgBasis))
        $specs.Add((Spec (Join-Path $exportRoot "3.9-金佩佩-2.txt") "$($person.Id)_ppg_channel_2.txt" "ppg" "high" $ppgBasis))
    }
    if ($person.Subject -eq "gww") {
        $gwwEeg = Join-Path $openBciRoot "OpenBCI-RAW-2025-03-09_14-15-41.txt"
        $gwwBasis = "EEG has 152,814 rows at 500 Hz (305.6 s); GWW transcript ends at 307 s. Duration difference is 1.4 s, and both are on 2025-03-09."
        $specs.Add((Spec $gwwEeg "$($person.Id)_eeg_raw.txt" "eeg" "high" $gwwBasis))
    }
    New-SupplementalSession $person.Id $person.Subject $person.Name @($specs) ([pscustomobject]@{
        status = "time_paired_supplemental_subject"
        confidence = if ($person.Subject -in @("gww", "jinpeipei", "yjy")) { "high_or_mixed" } else { "identity_only" }
        note = "Only modalities with defensible participant evidence are assigned as primary."
    })
}

# Keep ambiguous acquisitions inside the requested raw dataset without silently assigning them.
$unresolvedRoot = Join-Path $target "_unresolved_sensor_candidates"
New-Item -ItemType Directory -Force -Path $unresolvedRoot | Out-Null

$march8Root = Join-Path $unresolvedRoot "2025-03-08_eeg"
New-Item -ItemType Directory -Force -Path $march8Root | Out-Null
$march8Files = Get-ChildItem -LiteralPath $openBciRoot -File -Filter "OpenBCI-RAW-2025-03-08_*.txt" | Where-Object Length -ge 1MB
foreach ($file in $march8Files) {
    $destination = Join-Path $march8Root $file.Name
    Copy-Verified $file.FullName $destination
    Add-Audit "UNRESOLVED" "lzy|whz|wq|dyn|fzc|unknown" "EEG" "unresolved_candidate" "low" $file.FullName $destination "Same acquisition date, but 11 valid EEG files exist for 5 named subjects and clocks differ by roughly nine hours; no unique one-to-one mapping is defensible."
}

$march9Unresolved = Join-Path $openBciRoot "OpenBCI-RAW-2025-03-09_15-19-01.txt"
$march9Destination = Join-Path $unresolvedRoot "OpenBCI-RAW-2025-03-09_15-19-01.txt"
Copy-Verified $march9Unresolved $march9Destination
Add-Audit "UNRESOLVED" "march9_subject_unknown" "EEG" "unresolved_candidate" "low" $march9Unresolved $march9Destination "Valid 848.4 s EEG, but no participant has a matching name or sufficiently distinctive duration."

$faceUnresolved = Join-Path $exportRoot "action_info_20250309_105658.csv"
$faceDestination = Join-Path $unresolvedRoot "action_info_20250309_105658.csv"
Copy-Verified $faceUnresolved $faceDestination
Add-Audit "UNRESOLVED" "jinpeipei|ssg|li_unknown" "facial_actions" "unresolved_candidate" "low" $faceUnresolved $faceDestination "Its 10:56:58-11:09:21 interval is temporally near multiple DOCX creation times; identity cannot be uniquely determined."

$extraPpgRoot = Join-Path $unresolvedRoot "2025-03-05_ppg_extra_attempts"
New-Item -ItemType Directory -Force -Path $extraPpgRoot | Out-Null
foreach ($stamp in @("20-16-19", "20-18-07", "22-05-19", "22-09-18")) {
    foreach ($channel in 2, 3) {
        $source = Join-Path $ppgDeviceRoot "03-05-$($stamp)_ch$channel.txt"
        if ((Get-Item -LiteralPath $source).Length -eq 0) { continue }
        $destination = Join-Path $extraPpgRoot (Split-Path -Leaf $source)
        Copy-Verified $source $destination
        Add-Audit "UNRESOLVED" "hsy|ywj|device_test" "PPG" "unresolved_candidate" "low" $source $destination "Same-day extra acquisition without participant name; retained but not used as a primary session input."
    }
}

$rootManifest | Add-Member -NotePropertyName time_pairing_revision -NotePropertyValue ([pscustomobject]@{
    updated_at = (Get-Date).ToString("o")
    total_raw_sessions = @($rootManifest.sessions).Count
    original_processed_sessions = 18
    processed_data_stale = $true
    note = "Sessions 019-029 and recovered modalities require preprocessing before training. Ambiguous acquisitions are isolated under _unresolved_sensor_candidates."
}) -Force
Write-Json $rootManifestPath $rootManifest
Write-Json $sessionIndexPath $sessionIndex

$complianceJsonPath = Join-Path $target "session_compliance.json"
if (Test-Path -LiteralPath $complianceJsonPath) {
    $compliance = Get-Content -LiteralPath $complianceJsonPath -Raw | ConvertFrom-Json
    $compliance | Add-Member -NotePropertyName stale_after_time_pairing_revision -NotePropertyValue $true -Force
    $compliance | Add-Member -NotePropertyName report_scope_sessions -NotePropertyValue 18 -Force
    $compliance | Add-Member -NotePropertyName current_raw_sessions -NotePropertyValue @($rootManifest.sessions).Count -Force
    $compliance | Add-Member -NotePropertyName stale_reason -NotePropertyValue "Sessions 019-029 and recovered modalities have not been reprocessed or re-audited." -Force
    Write-Json $complianceJsonPath $compliance
}
$complianceMarkdownPath = Join-Path $target "session_compliance.md"
if (Test-Path -LiteralPath $complianceMarkdownPath) {
    $oldComplianceMarkdown = Get-Content -LiteralPath $complianceMarkdownPath -Raw
    if ($oldComplianceMarkdown -notmatch 'TIME-PAIRING REVISION WARNING') {
        $warning = "> **TIME-PAIRING REVISION WARNING:** This report covers the original 18 sessions only. The raw directory now contains 29 sessions plus recovered modalities; re-run preprocessing and compliance auditing before formal experiments.`r`n`r`n"
        ($warning + $oldComplianceMarkdown) | Set-Content -LiteralPath $complianceMarkdownPath -Encoding UTF8
    }
}

$auditPath = Join-Path $target "pairing_audit.json"
Write-Json $auditPath @($audit)
$audit | Export-Csv -LiteralPath (Join-Path $target "pairing_audit.csv") -NoTypeInformation -Encoding UTF8

$status = @"
# Time-paired raw dataset status

The requested target is the existing directory `code/data/raw/bci_subjects_ept_v1` (the path written with escaped underscores was interpreted as this directory).

- Original sessions retained: 18
- Supplemental 2025-03-09 subject sessions: 11 (`session_019` through `session_029`)
- Total raw subject sessions: 29
- HSY primary PPG corrected to the 21:03:04 pair; the earlier 20:56:09 short pair is preserved with explicit candidate filenames.
- ZHJ and ZYL direct-name EEG recordings added to their existing sessions.
- GWW EEG assigned by a 305.6 s versus 307 s duration match.
- Jin Peipei's direct-name two-file PPG pair added.
- Ambiguous March 8 EEG, one March 9 EEG, one March 9 face stream, and extra March 5 PPG attempts are retained under `_unresolved_sensor_candidates` and are not silently assigned.

Important: the existing processed dataset and frozen split still cover the original 18 sessions. They are intentionally not overwritten. Re-run preprocessing and define a new split before using sessions 019-029 or the recovered primary modalities in a formal experiment.
"@
$status | Set-Content -LiteralPath (Join-Path $target "TIME_PAIRING_STATUS.md") -Encoding UTF8

[pscustomobject]@{
    Target = $target
    RawSessions = @($rootManifest.sessions).Count
    AuditRows = $audit.Count
    UnresolvedFiles = (Get-ChildItem -LiteralPath $unresolvedRoot -Recurse -File).Count
    ProcessedDataStale = $true
    PairingAudit = $auditPath
} | Format-List
