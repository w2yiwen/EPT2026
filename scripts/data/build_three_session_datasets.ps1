[CmdletBinding()]
param(
    [string]$SourceRoot = "",
    [string]$OutputBase = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
    $SourceRoot = Join-Path $ProjectRoot "data\raw\bci_subjects_ept_v1"
}
if ([string]::IsNullOrWhiteSpace($OutputBase)) {
    $OutputBase = Join-Path $ProjectRoot "data\raw"
}

function Write-JsonFile {
    param(
        [Parameter(Mandatory)] [object]$Value,
        [Parameter(Mandatory)] [string]$Path,
        [int]$Depth = 20
    )
    $json = $Value | ConvertTo-Json -Depth $Depth
    Set-Content -LiteralPath $Path -Value $json -Encoding utf8
}

function Get-SessionState {
    param([Parameter(Mandatory)] [System.IO.DirectoryInfo]$SessionDirectory)

    $dataFiles = @(Get-ChildItem -LiteralPath $SessionDirectory.FullName -File | Where-Object {
        $_.Name -notin @("session_metadata.json", "source_manifest.json")
    })
    $prefix = [regex]::Escape($SessionDirectory.Name)
    $annotation = @($dataFiles | Where-Object { $_.Name -match "^${prefix}_annotations\.docx$" })
    $video = @($dataFiles | Where-Object { $_.Name -match "^${prefix}_video.*\.(mp4|avi|mov|mkv|mts|m2ts|wmv)$" })
    $audio = @($dataFiles | Where-Object { $_.Name -match "^${prefix}_audio.*\.(wav|mp3|m4a|flac|aac)$" })
    $eeg = @($dataFiles | Where-Object { $_.Name -match "^${prefix}_eeg_raw\.txt$" })
    $ppg1 = @($dataFiles | Where-Object { $_.Name -match "^${prefix}_ppg_channel_1\.txt$" })
    $ppg2 = @($dataFiles | Where-Object { $_.Name -match "^${prefix}_ppg_channel_2\.txt$" })
    $facial = @($dataFiles | Where-Object { $_.Name -match "^${prefix}_facial_actions\.csv$" })

    [pscustomobject]@{
        SessionId = $SessionDirectory.Name
        Directory = $SessionDirectory
        DataFiles = $dataFiles
        HasDocx = ($annotation.Count -gt 0)
        HasVideo = ($video.Count -gt 0)
        HasAudio = ($audio.Count -gt 0)
        HasEeg = ($eeg.Count -gt 0)
        HasCompletePpg = (($ppg1.Count -gt 0) -and ($ppg2.Count -gt 0))
        HasFacialActions = ($facial.Count -gt 0)
    }
}

function Test-IncludedFile {
    param(
        [Parameter(Mandatory)] [System.IO.FileInfo]$File,
        [Parameter(Mandatory)] [ValidateSet("all", "docx_video", "docx_video_physio")] [string]$Tier,
        [Parameter(Mandatory)] [string]$SessionId
    )

    if ($Tier -eq "all") { return $true }

    $prefix = [regex]::Escape($SessionId)
    $isDocx = $File.Extension -ieq ".docx"
    $isVideo = $File.Name -match "^${prefix}_video.*\.(mp4|avi|mov|mkv|mts|m2ts|wmv)$"
    $isAudio = $File.Name -match "^${prefix}_audio.*\.(wav|mp3|m4a|flac|aac)$"
    if ($isDocx -or $isVideo -or $isAudio) { return $true }

    if ($Tier -eq "docx_video_physio") {
        return (
            $File.Name -match "^${prefix}_eeg_raw\.txt$" -or
            $File.Name -match "^${prefix}_ppg_channel_[12]\.txt$" -or
            $File.Name -match "^${prefix}_facial_actions\.csv$"
        )
    }

    return $false
}

function Add-LinkedFile {
    param(
        [Parameter(Mandatory)] [string]$Source,
        [Parameter(Mandatory)] [string]$Destination
    )

    if (Test-Path -LiteralPath $Destination) {
        $sourceItem = Get-Item -LiteralPath $Source
        $destinationItem = Get-Item -LiteralPath $Destination
        if ($sourceItem.Length -ne $destinationItem.Length) {
            throw "Existing output has a different size: $Destination"
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

if (-not (Test-Path -LiteralPath $SourceRoot -PathType Container)) {
    throw "Source dataset does not exist: $SourceRoot"
}

$sourceIndexPath = Join-Path $SourceRoot "session_index.json"
if (-not (Test-Path -LiteralPath $sourceIndexPath -PathType Leaf)) {
    throw "Source session index is missing: $sourceIndexPath"
}

$sourceIndex = Get-Content -LiteralPath $sourceIndexPath -Raw | ConvertFrom-Json
$sourceIndexById = @{}
foreach ($entry in $sourceIndex.sessions) { $sourceIndexById[$entry.session_id] = $entry }

$sessionStates = @(Get-ChildItem -LiteralPath $SourceRoot -Directory | Where-Object {
    $_.Name -match '^session_\d+$'
} | Sort-Object Name | ForEach-Object { Get-SessionState -SessionDirectory $_ })

$definitions = @(
    [pscustomobject]@{
        Dataset = "bci_sessions_all_v1"
        Tier = "all"
        Title = "全部 session 数据集"
        Description = "包含源总库中的全部 session，并保留每个 session 当前已有的全部数据文件。"
    },
    [pscustomobject]@{
        Dataset = "bci_sessions_docx_video_v1"
        Tier = "docx_video"
        Title = "DOCX 与视频配对数据集"
        Description = "仅包含同时具备标注 DOCX 和视频的 session；目录中只保留 DOCX、视频和对应音频。"
    },
    [pscustomobject]@{
        Dataset = "bci_sessions_docx_video_physio_v1"
        Tier = "docx_video_physio"
        Title = "DOCX 视频与生理信号配对数据集"
        Description = "仅包含同时具备标注 DOCX、视频，以及 EEG 或完整双通道 PPG 的 session；同时保留可用面部动作信号。"
    }
)

$results = @()
$generatedAt = (Get-Date).ToString("o")

foreach ($definition in $definitions) {
    $datasetRoot = Join-Path $OutputBase $definition.Dataset
    New-Item -ItemType Directory -Path $datasetRoot -Force | Out-Null

    $selected = @($sessionStates | Where-Object {
        if ($definition.Tier -eq "all") { return $true }
        if ($definition.Tier -eq "docx_video") { return ($_.HasDocx -and $_.HasVideo) }
        return ($_.HasDocx -and $_.HasVideo -and ($_.HasEeg -or $_.HasCompletePpg))
    })

    $datasetFiles = @()
    $matrix = @()
    $indexEntries = @()
    $hardLinks = 0
    $copies = 0
    $existing = 0
    $hardLinkFiles = 0
    $copyFiles = 0

    foreach ($state in $selected) {
        $sessionDestination = Join-Path $datasetRoot $state.SessionId
        New-Item -ItemType Directory -Path $sessionDestination -Force | Out-Null

        $includedFiles = @($state.DataFiles | Where-Object {
            Test-IncludedFile -File $_ -Tier $definition.Tier -SessionId $state.SessionId
        })

        foreach ($file in $includedFiles) {
            $destination = Join-Path $sessionDestination $file.Name
            $mode = Add-LinkedFile -Source $file.FullName -Destination $destination
            if ($mode -eq "hardlink") { $hardLinks++ }
            elseif ($mode -eq "copy") { $copies++ }
            else { $existing++ }
            $destinationItem = Get-Item -LiteralPath $destination
            if ($destinationItem.LinkType -eq "HardLink") { $hardLinkFiles++ }
            else { $copyFiles++ }
        }

        $sourceMetadataPath = Join-Path $state.Directory.FullName "session_metadata.json"
        $sourceMetadata = Get-Content -LiteralPath $sourceMetadataPath -Raw | ConvertFrom-Json
        $includedNames = @{}
        foreach ($file in $includedFiles) { $includedNames[$file.Name] = $true }
        $fileRecords = @($sourceMetadata.files | Where-Object {
            $leaf = Split-Path -Leaf $_.staged_relative_path
            $includedNames.ContainsKey($leaf)
        })

        foreach ($file in $includedFiles) {
            $hasRecord = @($fileRecords | Where-Object {
                (Split-Path -Leaf $_.staged_relative_path) -eq $file.Name
            }).Count -gt 0
            if (-not $hasRecord) {
                $fileRecords += [pscustomobject]@{
                    session_id = $state.SessionId
                    subject_id = $sourceMetadata.subject_id
                    source_absolute_path = $file.FullName
                    staged_relative_path = "$($state.SessionId)/$($file.Name)"
                    bytes = $file.Length
                    sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
                }
            }
        }

        $availableSources = [ordered]@{
            eeg = (@($includedFiles | Where-Object Name -match '_eeg_raw\.txt$').Count -gt 0)
            paired_ppg = (
                @($includedFiles | Where-Object Name -match '_ppg_channel_1\.txt$').Count -gt 0 -and
                @($includedFiles | Where-Object Name -match '_ppg_channel_2\.txt$').Count -gt 0
            )
            facial_actions = (@($includedFiles | Where-Object Name -match '_facial_actions\.csv$').Count -gt 0)
            openface_features = $false
            audio = (@($includedFiles | Where-Object Name -match '_audio.*\.(wav|mp3|m4a|flac|aac)$').Count -gt 0)
            video = (@($includedFiles | Where-Object Name -match '_video.*\.(mp4|avi|mov|mkv|mts|m2ts|wmv)$').Count -gt 0)
            annotated_transcript = (@($includedFiles | Where-Object Name -match '_annotations\.docx$').Count -gt 0)
        }

        $sessionMetadata = [ordered]@{
            dataset = $definition.Dataset
            source_dataset = "bci_subjects_ept_v1"
            dataset_tier = $definition.Tier
            session_id = $state.SessionId
            subject_id = $sourceMetadata.subject_id
            annotation_mode = $sourceMetadata.annotation_mode
            available_sources = $availableSources
            files = $fileRecords
            assembly = [ordered]@{
                generated_at = $generatedAt
                source_session = $state.Directory.FullName
                storage = "NTFS hard links when supported; byte copies only as fallback"
            }
        }
        Write-JsonFile -Value $sessionMetadata -Path (Join-Path $sessionDestination "session_metadata.json")
        Write-JsonFile -Value $sessionMetadata -Path (Join-Path $sessionDestination "source_manifest.json")

        $logicalBytes = ($includedFiles | Measure-Object -Property Length -Sum).Sum
        $matrix += [pscustomobject]@{
            session_id = $state.SessionId
            subject_id = $sourceMetadata.subject_id
            docx = $availableSources.annotated_transcript
            video = $availableSources.video
            audio = $availableSources.audio
            eeg = $availableSources.eeg
            paired_ppg = $availableSources.paired_ppg
            facial_actions = $availableSources.facial_actions
            file_count = $includedFiles.Count
            logical_bytes = $logicalBytes
        }

        if ($sourceIndexById.ContainsKey($state.SessionId)) {
            $old = $sourceIndexById[$state.SessionId]
            $indexEntries += [pscustomobject]@{
                session_id = $old.session_id
                subject_id = $old.subject_id
                split = $old.split
                raw_path = "raw/$($definition.Dataset)/$($state.SessionId)"
                processed_path = $null
                source_raw_path = $old.raw_path
            }
        }
        else {
            $indexEntries += [pscustomobject]@{
                session_id = $state.SessionId
                subject_id = $sourceMetadata.subject_id
                split = "unassigned"
                raw_path = "raw/$($definition.Dataset)/$($state.SessionId)"
                processed_path = $null
                source_raw_path = "raw/bci_subjects_ept_v1/$($state.SessionId)"
            }
        }

        $datasetFiles += $fileRecords
    }

    $indexObject = [ordered]@{
        dataset = $definition.Dataset
        source_dataset = "bci_subjects_ept_v1"
        selection_tier = $definition.Tier
        sessions = $indexEntries
    }
    Write-JsonFile -Value $indexObject -Path (Join-Path $datasetRoot "session_index.json")

    $manifestObject = [ordered]@{
        dataset = $definition.Dataset
        source_dataset = "bci_subjects_ept_v1"
        generated_at = $generatedAt
        selection_tier = $definition.Tier
        session_count = $selected.Count
        logical_file_count = ($matrix.file_count | Measure-Object -Sum).Sum
        logical_bytes = ($matrix.logical_bytes | Measure-Object -Sum).Sum
        storage = [ordered]@{
            hard_link_files = $hardLinkFiles
            fallback_copy_files = $copyFiles
            hard_links_created = $hardLinks
            fallback_copies_created = $copies
            existing_files_reused = $existing
        }
        files = $datasetFiles
    }
    Write-JsonFile -Value $manifestObject -Path (Join-Path $datasetRoot "source_manifest.json")
    Write-JsonFile -Value $manifestObject -Path (Join-Path $datasetRoot "dataset_manifest.json")

    $matrix | Export-Csv -LiteralPath (Join-Path $datasetRoot "completeness_matrix.csv") -NoTypeInformation -Encoding utf8

    $readme = @"
# $($definition.Title)

$($definition.Description)

- 源数据集：$SourceRoot
- 生成时间：$generatedAt
- session 数：$($selected.Count)
- session 编号：$((@($selected.SessionId) -join ', '))
- 逻辑文件数：$(($matrix.file_count | Measure-Object -Sum).Sum)
- 逻辑数据量：$(($matrix.logical_bytes | Measure-Object -Sum).Sum) bytes
- 存储方式：优先使用 NTFS 硬链接，避免三版重复占用空间；文件内容与源文件逐字节一致。
- 处理状态：本目录是 raw 数据集，`processed_path` 留空；正式实验前应按本版清单重新预处理并重新划分 split。

筛选规则：

- all：全部 session，保留其当前所有数据文件。
- docx_video：要求标注 DOCX 和视频同时存在，只保留 DOCX、视频及音频。
- docx_video_physio：在 DOCX 与视频同时存在的基础上，要求至少有 EEG 或完整的 PPG 双通道；同时保留面部动作 CSV。

详细模态完整性见 completeness_matrix.csv，来源与哈希见 source_manifest.json。
"@
    Set-Content -LiteralPath (Join-Path $datasetRoot "README.md") -Value $readme -Encoding utf8

    $results += [pscustomobject]@{
        Dataset = $definition.Dataset
        Sessions = $selected.Count
        Files = ($matrix.file_count | Measure-Object -Sum).Sum
        LogicalBytes = ($matrix.logical_bytes | Measure-Object -Sum).Sum
        HardLinkFiles = $hardLinkFiles
        CopyFiles = $copyFiles
        HardLinksCreated = $hardLinks
        FallbackCopies = $copies
        ExistingFiles = $existing
        Path = $datasetRoot
    }
}

$summaryObject = [ordered]@{
    generated_at = $generatedAt
    source_dataset = $SourceRoot
    output_base = $OutputBase
    datasets = $results
}
Write-JsonFile -Value $summaryObject -Path (Join-Path $OutputBase "three_dataset_versions.json")

$summaryMarkdown = @"
# 三版 session 数据集索引

三版数据均从 $SourceRoot 生成，保留原 session 编号。数据文件使用 NTFS 硬链接，三版不会重复占用大文件空间；各版生成的 JSON、CSV 和 README 为独立文件。

## 第一版 全部 session

- 路径：$(Join-Path $OutputBase 'bci_sessions_all_v1')
- session：$($results[0].Sessions)
- 数据文件：$($results[0].Files)
- 规则：保留 41 个 session 当前已有的全部数据文件。

## 第二版 DOCX 与视频

- 路径：$(Join-Path $OutputBase 'bci_sessions_docx_video_v1')
- session：$($results[1].Sessions)
- 数据文件：$($results[1].Files)
- 规则：只收录同时具备标注 DOCX 和视频的 session；目录内只保留 DOCX、视频和音频。

## 第三版 DOCX 视频与生理信号

- 路径：$(Join-Path $OutputBase 'bci_sessions_docx_video_physio_v1')
- session：$($results[2].Sessions)
- 数据文件：$($results[2].Files)
- 规则：在第二版配对基础上，要求至少具备 EEG 或完整双通道 PPG，并保留可用面部动作 CSV。

每一版都包含 session_index.json、dataset_manifest.json、source_manifest.json 和 completeness_matrix.csv。正式实验前应分别运行预处理并重新生成 split。
"@
Set-Content -LiteralPath (Join-Path $OutputBase "THREE_DATASET_VERSIONS.md") -Value $summaryMarkdown -Encoding utf8

$results | Format-Table -AutoSize
