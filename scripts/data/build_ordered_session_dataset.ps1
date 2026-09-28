[CmdletBinding()]
param(
    [string]$SourceRoot = "",
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
    $SourceRoot = Join-Path $ProjectRoot "data\raw\bci_subjects_ept_v1"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $ProjectRoot "data\raw\bci_sessions_v1"
}

function Write-JsonFile {
    param(
        [Parameter(Mandatory)] [object]$Value,
        [Parameter(Mandatory)] [string]$Path,
        [int]$Depth = 24
    )
    $Value | ConvertTo-Json -Depth $Depth | Set-Content -LiteralPath $Path -Encoding utf8
}

function Add-HardLinkOrCopy {
    param(
        [Parameter(Mandatory)] [string]$Source,
        [Parameter(Mandatory)] [string]$Destination
    )

    if (Test-Path -LiteralPath $Destination) {
        $sourceItem = Get-Item -LiteralPath $Source
        $destinationItem = Get-Item -LiteralPath $Destination
        if ($sourceItem.Length -ne $destinationItem.Length) {
            throw "Existing output differs in size: $Destination"
        }
        return "existing"
    }

    try {
        New-Item -ItemType HardLink -Path $Destination -Target $Source | Out-Null
        return "hardlink"
    }
    catch {
        Copy-Item -LiteralPath $Source -Destination $Destination
        return "copy"
    }
}

function Get-SourceState {
    param([Parameter(Mandatory)] [System.IO.DirectoryInfo]$Directory)

    $files = @(Get-ChildItem -LiteralPath $Directory.FullName -File | Where-Object {
        $_.Name -notin @("session_metadata.json", "source_manifest.json")
    })
    $prefix = [regex]::Escape($Directory.Name)
    [pscustomobject]@{
        OldSessionId = $Directory.Name
        Directory = $Directory
        Files = $files
        HasDocx = (@($files | Where-Object Name -match "^${prefix}_annotations\.docx$").Count -gt 0)
        HasVideo = (@($files | Where-Object Name -match "^${prefix}_video.*\.(mp4|avi|mov|mkv|mts|m2ts|wmv)$").Count -gt 0)
        HasAudio = (@($files | Where-Object Name -match "^${prefix}_audio.*\.(wav|mp3|m4a|flac|aac)$").Count -gt 0)
        HasEeg = (@($files | Where-Object Name -match "^${prefix}_eeg_raw\.txt$").Count -gt 0)
        HasPpg1 = (@($files | Where-Object Name -match "^${prefix}_ppg_channel_1\.txt$").Count -gt 0)
        HasPpg2 = (@($files | Where-Object Name -match "^${prefix}_ppg_channel_2\.txt$").Count -gt 0)
        HasFacial = (@($files | Where-Object Name -match "^${prefix}_facial_actions\.csv$").Count -gt 0)
    }
}

if (-not (Test-Path -LiteralPath $SourceRoot -PathType Container)) {
    throw "Source dataset does not exist: $SourceRoot"
}

$sourceIndexPath = Join-Path $SourceRoot "session_index.json"
if (-not (Test-Path -LiteralPath $sourceIndexPath -PathType Leaf)) {
    throw "Missing source index: $sourceIndexPath"
}

$sourceIndex = Get-Content -LiteralPath $sourceIndexPath -Raw | ConvertFrom-Json
$sourceIndexById = @{}
foreach ($entry in $sourceIndex.sessions) { $sourceIndexById[$entry.session_id] = $entry }

$states = @(Get-ChildItem -LiteralPath $SourceRoot -Directory | Where-Object {
    $_.Name -match '^session_\d+$'
} | Sort-Object Name | ForEach-Object { Get-SourceState -Directory $_ })

if ($states.Count -ne 41) {
    throw "Expected 41 source sessions, found $($states.Count)."
}

$priorityComplete = @(
    "session_004",
    "session_006",
    "session_011",
    "session_012",
    "session_015",
    "session_017",
    "session_018"
)

$docxVideoAll = @($states | Where-Object { $_.HasDocx -and $_.HasVideo } | Select-Object -ExpandProperty OldSessionId)
if ($docxVideoAll.Count -ne 12) {
    throw "Expected 12 DOCX+video sessions, found $($docxVideoAll.Count)."
}

foreach ($sessionId in $priorityComplete) {
    if ($sessionId -notin $docxVideoAll) {
        throw "Priority session is not in the DOCX+video set: $sessionId"
    }
    $state = $states | Where-Object OldSessionId -eq $sessionId
    if (-not ($state.HasEeg -or ($state.HasPpg1 -and $state.HasPpg2))) {
        throw "Priority session lacks EEG and complete PPG: $sessionId"
    }
}

$priorityDocxVideo = @($docxVideoAll | Where-Object { $_ -notin $priorityComplete } | Sort-Object)
$priorityOther = @($states.OldSessionId | Where-Object {
    $_ -notin $priorityComplete -and $_ -notin $priorityDocxVideo
} | Sort-Object)
$orderedOldIds = @($priorityComplete + $priorityDocxVideo + $priorityOther)

if (($orderedOldIds | Sort-Object -Unique).Count -ne 41) {
    throw "Ordered session list is not a unique 41-session permutation."
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$datasetName = Split-Path -Leaf $OutputRoot
$generatedAt = (Get-Date).ToString("o")
$mapping = @()
$matrix = @()
$indexEntries = @()
$aggregateRecords = @()
$hardLinksCreated = 0
$copiesCreated = 0
$existingFiles = 0
$hardLinkFiles = 0
$copyFiles = 0

for ($i = 0; $i -lt $orderedOldIds.Count; $i++) {
    $oldId = $orderedOldIds[$i]
    $newId = "session_{0:D3}" -f ($i + 1)
    $state = $states | Where-Object OldSessionId -eq $oldId
    $oldIndex = $sourceIndexById[$oldId]
    $group = if ($oldId -in $priorityComplete) {
        "complete_multimodal"
    }
    elseif ($oldId -in $priorityDocxVideo) {
        "docx_video"
    }
    else {
        "other"
    }

    $destinationDirectory = Join-Path $OutputRoot $newId
    New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
    $renamedFiles = @()

    foreach ($sourceFile in $state.Files) {
        if ($sourceFile.Name -notmatch "^$([regex]::Escape($oldId))") {
            throw "Non-standard source filename in ${oldId}: $($sourceFile.Name)"
        }
        $newName = $sourceFile.Name -replace "^$([regex]::Escape($oldId))", $newId
        $destinationPath = Join-Path $destinationDirectory $newName
        $mode = Add-HardLinkOrCopy -Source $sourceFile.FullName -Destination $destinationPath
        if ($mode -eq "hardlink") { $hardLinksCreated++ }
        elseif ($mode -eq "copy") { $copiesCreated++ }
        else { $existingFiles++ }
        $destinationItem = Get-Item -LiteralPath $destinationPath
        if ($destinationItem.LinkType -eq "HardLink") { $hardLinkFiles++ }
        else { $copyFiles++ }
        $renamedFiles += [pscustomobject]@{
            SourceFile = $sourceFile
            NewName = $newName
            DestinationPath = $destinationPath
        }
    }

    $sourceMetadataPath = Join-Path $state.Directory.FullName "session_metadata.json"
    $sourceMetadata = Get-Content -LiteralPath $sourceMetadataPath -Raw | ConvertFrom-Json
    $records = @()
    foreach ($renamed in $renamedFiles) {
        $oldName = $renamed.SourceFile.Name
        $sourceRecord = @($sourceMetadata.files | Where-Object {
            (Split-Path -Leaf $_.staged_relative_path) -eq $oldName
        } | Select-Object -First 1)

        if ($sourceRecord.Count -gt 0) {
            $record = [ordered]@{}
            foreach ($property in $sourceRecord[0].PSObject.Properties) {
                $record[$property.Name] = $property.Value
            }
            $record["session_id"] = $newId
            $record["original_session_id"] = $oldId
            $record["staged_relative_path"] = "$newId/$($renamed.NewName)"
        }
        else {
            $record = [ordered]@{
                session_id = $newId
                original_session_id = $oldId
                subject_id = $oldIndex.subject_id
                source_absolute_path = $renamed.SourceFile.FullName
                staged_relative_path = "$newId/$($renamed.NewName)"
                bytes = $renamed.SourceFile.Length
                sha256 = (Get-FileHash -LiteralPath $renamed.SourceFile.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            }
        }
        $records += [pscustomobject]$record
    }

    $availableSources = [ordered]@{
        eeg = $state.HasEeg
        paired_ppg = ($state.HasPpg1 -and $state.HasPpg2)
        facial_actions = $state.HasFacial
        openface_features = $false
        audio = $state.HasAudio
        video = $state.HasVideo
        annotated_transcript = $state.HasDocx
    }

    $sessionMetadata = [ordered]@{
        dataset = $datasetName
        source_dataset = "bci_subjects_ept_v1"
        session_id = $newId
        original_session_id = $oldId
        subject_id = $oldIndex.subject_id
        priority_group = $group
        annotation_mode = $sourceMetadata.annotation_mode
        available_sources = $availableSources
        files = $records
        assembly = [ordered]@{
            generated_at = $generatedAt
            source_session = $state.Directory.FullName
            storage = "NTFS hard links when supported; byte copies only as fallback"
        }
    }
    Write-JsonFile -Value $sessionMetadata -Path (Join-Path $destinationDirectory "session_metadata.json")
    Write-JsonFile -Value $sessionMetadata -Path (Join-Path $destinationDirectory "source_manifest.json")

    $logicalBytes = ($state.Files | Measure-Object -Property Length -Sum).Sum
    $mapping += [pscustomobject]@{
        new_session_id = $newId
        original_session_id = $oldId
        subject_id = $oldIndex.subject_id
        priority_group = $group
        legacy_split = $oldIndex.split
    }
    $matrix += [pscustomobject]@{
        session_id = $newId
        original_session_id = $oldId
        subject_id = $oldIndex.subject_id
        priority_group = $group
        docx = $state.HasDocx
        video = $state.HasVideo
        audio = $state.HasAudio
        eeg = $state.HasEeg
        paired_ppg = ($state.HasPpg1 -and $state.HasPpg2)
        facial_actions = $state.HasFacial
        file_count = $state.Files.Count
        logical_bytes = $logicalBytes
    }
    $indexEntries += [pscustomobject]@{
        session_id = $newId
        original_session_id = $oldId
        subject_id = $oldIndex.subject_id
        priority_group = $group
        split = "unassigned"
        legacy_split = $oldIndex.split
        raw_path = "raw/$datasetName/$newId"
        processed_path = $null
        source_raw_path = $oldIndex.raw_path
    }
    $aggregateRecords += $records
}

$mapping | Export-Csv -LiteralPath (Join-Path $OutputRoot "session_mapping.csv") -NoTypeInformation -Encoding utf8
$matrix | Export-Csv -LiteralPath (Join-Path $OutputRoot "completeness_matrix.csv") -NoTypeInformation -Encoding utf8

$indexObject = [ordered]@{
    dataset = $datasetName
    source_dataset = "bci_subjects_ept_v1"
    generated_at = $generatedAt
    ordering = @("complete_multimodal", "docx_video", "other")
    sessions = $indexEntries
}
Write-JsonFile -Value $indexObject -Path (Join-Path $OutputRoot "session_index.json")

$manifestObject = [ordered]@{
    dataset = $datasetName
    source_dataset = "bci_subjects_ept_v1"
    generated_at = $generatedAt
    session_count = $mapping.Count
    group_counts = [ordered]@{
        complete_multimodal = @($mapping | Where-Object priority_group -eq "complete_multimodal").Count
        docx_video = @($mapping | Where-Object priority_group -eq "docx_video").Count
        other = @($mapping | Where-Object priority_group -eq "other").Count
    }
    data_file_count = ($matrix.file_count | Measure-Object -Sum).Sum
    logical_bytes = ($matrix.logical_bytes | Measure-Object -Sum).Sum
    storage = [ordered]@{
        hard_link_files = $hardLinkFiles
        fallback_copy_files = $copyFiles
        hard_links_created_this_run = $hardLinksCreated
        fallback_copies_created_this_run = $copiesCreated
        existing_files_reused = $existingFiles
    }
    mapping_file = "session_mapping.csv"
    files = $aggregateRecords
}
Write-JsonFile -Value $manifestObject -Path (Join-Path $OutputRoot "dataset_manifest.json")
Write-JsonFile -Value $manifestObject -Path (Join-Path $OutputRoot "source_manifest.json")

$readme = @"
# BCI Sessions V1

这是重新编号后的完整 41-session raw 数据集。文件夹统一命名为 session_001 至 session_041，所有内部数据文件同步使用对应 session 前缀。

排序规则：

1. session_001 至 session_007：完整多模态优先组，对应原 session_004、006、011、012、015、017、018。
2. session_008 至 session_012：其余 DOCX 与视频配对组。完整多模态优先组本身属于 12 个 DOCX+视频 session，因此这里仅放未重复的其余 5 个。
3. session_013 至 session_041：其他 session，按原 session 编号升序排列。

重要说明：

- 原 session 与新 session 的一一对应关系见 session_mapping.csv。
- 所有新 split 均设为 unassigned；旧 split 仅保留在 legacy_split 字段，正式实验前应重新划分。
- 数据文件优先使用 NTFS 硬链接，不重复占用大文件空间。不要直接修改 raw 文件内容，因为硬链接与源总库共享同一文件内容。
- 模态完整性见 completeness_matrix.csv，来源与哈希见 source_manifest.json。
"@
Set-Content -LiteralPath (Join-Path $OutputRoot "README.md") -Value $readme -Encoding utf8

[pscustomobject]@{
    Dataset = $datasetName
    Sessions = $mapping.Count
    CompleteMultimodal = $manifestObject.group_counts.complete_multimodal
    DocxVideoRemaining = $manifestObject.group_counts.docx_video
    Other = $manifestObject.group_counts.other
    DataFiles = $manifestObject.data_file_count
    LogicalBytes = $manifestObject.logical_bytes
    HardLinkFiles = $hardLinkFiles
    FallbackCopies = $copyFiles
    OutputRoot = $OutputRoot
} | Format-List
