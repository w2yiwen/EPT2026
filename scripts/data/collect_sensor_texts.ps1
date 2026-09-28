param(
    [Parameter(Mandatory = $true)][string[]]$Roots,
    [string]$Destination = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
if (-not $Destination) {
    $Destination = Join-Path $projectRoot "BCI\sensor_text_collection_2025-03-05_to_03-08"
}
$destinationFull = [System.IO.Path]::GetFullPath($Destination)
$periodStart = [datetime]"2025-03-05"
$periodEnd = [datetime]"2025-03-09 23:59:59"

if (-not (Get-Command rg -ErrorAction SilentlyContinue)) {
    throw "ripgrep (rg) is required for the full-drive scan."
}
if (Test-Path -LiteralPath (Join-Path $destinationFull "manifest.json")) {
    throw "Destination already contains a completed collection: $destinationFull"
}

New-Item -ItemType Directory -Force -Path $destinationFull | Out-Null
$ppgRoot = New-Item -ItemType Directory -Force -Path (Join-Path $destinationFull "ppg_pairs")
$eegRoot = New-Item -ItemType Directory -Force -Path (Join-Path $destinationFull "eeg_sessions")
$eegShortRoot = New-Item -ItemType Directory -Force -Path (Join-Path $destinationFull "eeg_short_or_header_only_candidates")

$knownSubjects = @(
    "ajy", "dyn", "fzc", "hsy", "lfy", "lmx", "luxinyu", "lzy", "whz",
    "wq", "wyb", "xfx", "ypj", "ywj", "zhj", "zsr", "zyl", "zyx", "jpp"
)
$nameAliases = [ordered]@{
    "hesiyu" = "hsy"
    "limaoxi" = "lmx"
    "wangyibai" = "wyb"
    "xiefenxia" = "xfx"
    "zhanghejia" = "zhj"
    "zhaoyunlu" = "zyl"
    "zhaoyinxing" = "zyx"
}

function Get-SafeName([string]$Value) {
    $safe = $Value -replace '[<>:"/\\|?*]', "_"
    $safe = $safe.Trim().TrimEnd(".")
    if (-not $safe) { return "unknown" }
    return $safe
}

function Get-Subject([string]$Path, [string]$Name) {
    if ($Name.Contains("金佩佩")) { return "jpp" }
    $normalizedName = ($Name -replace '[^A-Za-z0-9]', "").ToLowerInvariant()
    foreach ($alias in $nameAliases.Keys) {
        if ($normalizedName.Contains($alias)) { return $nameAliases[$alias] }
    }
    $parent = Split-Path -Leaf (Split-Path -Parent $Path)
    if ($knownSubjects -contains $parent.ToLowerInvariant()) {
        return $parent.ToLowerInvariant()
    }
    $lowerName = $Name.ToLowerInvariant()
    foreach ($subject in $knownSubjects) {
        if ($lowerName -match "(^|[^a-z])$([regex]::Escape($subject))([^a-z]|$)") {
            return $subject
        }
    }
    return "unknown"
}

function Get-PpgStamp([string]$Name) {
    $match = [regex]::Match($Name, '(?i)(?<month>0?3)[-_](?<day>0?[5-9])[-_](?<hour>\d{2})[-_](?<minute>\d{2})[-_](?<second>\d{2})')
    if ($match.Success) {
        return "{0:D2}-{1:D2}-{2}-{3}-{4}" -f [int]$match.Groups['month'].Value, [int]$match.Groups['day'].Value, $match.Groups['hour'].Value, $match.Groups['minute'].Value, $match.Groups['second'].Value
    }
    if ($Name -match '(?i)^3[.-]9-.+-[12]\.txt$') { return "03-09-time-unknown" }
    return $null
}

function Get-EegStamp([string]$Name) {
    $match = [regex]::Match($Name, '(?<year>2025)[-_](?<month>0?3)[-_](?<day>0?[5-9])[_-](?<hour>\d{2})[-_](?<minute>\d{2})[-_](?<second>\d{2})')
    if (-not $match.Success) { return $null }
    return "{0}-{1:D2}-{2:D2}_{3}-{4}-{5}" -f $match.Groups['year'].Value, [int]$match.Groups['month'].Value, [int]$match.Groups['day'].Value, $match.Groups['hour'].Value, $match.Groups['minute'].Value, $match.Groups['second'].Value
}

function Get-PpgBase([string]$Name) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($Name)
    $stem = $stem -replace '(?i)[\s_-]*ch(?:annel)?[\s_-]*[23].*$', ''
    $stem = $stem -replace '(?i)[\s_-]*ppg[\s_-]*channel[\s_-]*[12].*$', ''
    if ($stem -match '(?i)^3[.-]9-.+-[12]$') { $stem = $stem -replace '-[12]$', '' }
    return $stem.Trim(' ', '-', '_')
}

function Test-ScalarSignal([string]$Path) {
    if ((Get-Item -LiteralPath $Path).Length -eq 0) { return $false }
    $lines = @(Get-Content -LiteralPath $Path -TotalCount 12 -ErrorAction SilentlyContinue | Where-Object { $_.Trim() })
    if ($lines.Count -lt 3) { return $false }
    $numeric = 0
    foreach ($line in $lines) {
        $value = 0.0
        if ([double]::TryParse($line.Trim(), [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$value)) {
            $numeric += 1
        }
    }
    return $numeric -ge [math]::Min(3, $lines.Count)
}

function Test-OpenBci([string]$Path) {
    $head = (Get-Content -LiteralPath $Path -TotalCount 6 -ErrorAction SilentlyContinue) -join "`n"
    return $head -match '%OpenBCI Raw EEG Data'
}

function Copy-WithUniqueName([string]$Source, [string]$TargetDirectory) {
    $name = [System.IO.Path]::GetFileName($Source)
    $target = Join-Path $TargetDirectory $name
    if (Test-Path -LiteralPath $target) {
        $sourceHash = (Get-FileHash -LiteralPath $Source -Algorithm SHA256).Hash
        $targetHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
        if ($sourceHash -eq $targetHash) { return $target }
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($name)
        $extension = [System.IO.Path]::GetExtension($name)
        $target = Join-Path $TargetDirectory ("{0}__{1}{2}" -f $stem, $sourceHash.Substring(0, 8), $extension)
    }
    Copy-Item -LiteralPath $Source -Destination $target
    return $target
}

$existingRoots = @($Roots | Where-Object { Test-Path -LiteralPath $_ })
$allTxt = @(& rg --files -uu -g '*.txt' @existingRoots 2>$null)
$allTxt = @($allTxt | Where-Object {
    $_ -and ([System.IO.Path]::GetFullPath($_) -notlike "$destinationFull*")
})

$ppgCandidates = @()
$eegCandidates = @()
foreach ($path in $allTxt) {
    $name = [System.IO.Path]::GetFileName($path)
    $ppgNameMatch = (
        $name -match '(?i)(ch(?:annel)?[\s_-]*[23]|ppg[\s_-]*channel[\s_-]*[12])' -or
        $name -match '(?i)^3[.-]9-.+-[12]\.txt$'
    )
    if ($ppgNameMatch) {
        $stamp = Get-PpgStamp $name
        if ($stamp -and (Test-ScalarSignal $path)) {
            $item = Get-Item -LiteralPath $path
            $ppgCandidates += [pscustomobject]@{
                Path = $item.FullName
                Directory = $item.DirectoryName
                Name = $item.Name
                Base = Get-PpgBase $item.Name
                Stamp = $stamp
                Subject = Get-Subject $item.FullName $item.Name
                Length = $item.Length
            }
        }
    }
    if ($name -match '(?i)(open[\s_-]*bci|eeg|脑电)' -and (Test-OpenBci $path)) {
        $stamp = Get-EegStamp $name
        if ($stamp) {
            $item = Get-Item -LiteralPath $path
            $eegCandidates += [pscustomobject]@{
                Path = $item.FullName
                Name = $item.Name
                Stamp = $stamp
                Subject = Get-Subject $item.FullName $item.Name
                Length = $item.Length
            }
        }
    }
}

$inventory = [System.Collections.Generic.List[object]]::new()
$unpaired = [System.Collections.Generic.List[object]]::new()
$seenPairHashes = @{}
$pairNumber = 0

foreach ($group in ($ppgCandidates | Group-Object Directory, Base | Sort-Object Name)) {
    $files = @($group.Group | Sort-Object Name)
    $ch2 = @($files | Where-Object { $_.Name -match '(?i)ch(?:annel)?[\s_-]*2' })
    $ch3 = @($files | Where-Object { $_.Name -match '(?i)ch(?:annel)?[\s_-]*3' })
    $numberedPair = @($files | Where-Object { $_.Name -match '(?i)^3[.-]9-.+-[12]\.txt$' })
    $pairs = @()
    if ($ch2.Count -ge 1 -and $ch3.Count -ge 1) {
        $limit = [math]::Min($ch2.Count, $ch3.Count)
        for ($index = 0; $index -lt $limit; $index++) {
            $pairs += ,@($ch2[$index], $ch3[$index])
        }
    } elseif ($ch2.Count -eq 2 -and $ch3.Count -eq 0) {
        # One acquisition saved its two optical channels as ch2 (1) and ch2 (2).
        $pairs += ,@($ch2[0], $ch2[1])
    } elseif ($numberedPair.Count -eq 2) {
        # The 2025-03-09 export used participant-name -1/-2 instead of ch2/ch3.
        $pairs += ,@($numberedPair[0], $numberedPair[1])
    }

    if ($pairs.Count -eq 0) {
        foreach ($file in $files) {
            $unpaired.Add([pscustomobject]@{ kind = "PPG"; path = $file.Path; reason = "no matching second channel" })
        }
        continue
    }

    foreach ($pair in $pairs) {
        $hash1 = (Get-FileHash -LiteralPath $pair[0].Path -Algorithm SHA256).Hash
        $hash2 = (Get-FileHash -LiteralPath $pair[1].Path -Algorithm SHA256).Hash
        $pairHash = (@($hash1, $hash2) | Sort-Object) -join ':'
        if ($seenPairHashes.ContainsKey($pairHash)) {
            $seenPairHashes[$pairHash].duplicate_sources += @($pair[0].Path, $pair[1].Path)
            continue
        }
        $pairNumber += 1
        $subject = @($pair | Where-Object Subject -ne "unknown" | Select-Object -First 1).Subject
        if (-not $subject) { $subject = "unknown" }
        $stamp = $pair[0].Stamp
        $folderName = Get-SafeName ("{0}__{1}" -f $subject, $stamp)
        $targetFolder = Join-Path $ppgRoot.FullName $folderName
        if (Test-Path -LiteralPath $targetFolder) {
            $targetFolder = "{0}__{1}" -f $targetFolder, $hash1.Substring(0, 8)
        }
        New-Item -ItemType Directory -Force -Path $targetFolder | Out-Null
        $copied1 = Copy-WithUniqueName $pair[0].Path $targetFolder
        $copied2 = Copy-WithUniqueName $pair[1].Path $targetFolder
        $record = [pscustomobject][ordered]@{
            kind = "PPG_PAIR"
            subject = $subject
            timestamp = $stamp
            destination_folder = $targetFolder
            destination_files = @($copied1, $copied2)
            source_files = @($pair[0].Path, $pair[1].Path)
            duplicate_sources = @()
            sha256 = @($hash1, $hash2)
            bytes = [int64]($pair[0].Length + $pair[1].Length)
        }
        $seenPairHashes[$pairHash] = $record
        $inventory.Add($record)
    }
}

$seenEegHashes = @{}
foreach ($candidate in ($eegCandidates | Sort-Object Subject, Stamp, Path)) {
    $hash = (Get-FileHash -LiteralPath $candidate.Path -Algorithm SHA256).Hash
    if ($seenEegHashes.ContainsKey($hash)) {
        $seenEegHashes[$hash].duplicate_sources += $candidate.Path
        continue
    }
    $kind = if ($candidate.Length -ge 1MB) { "EEG" } else { "EEG_SHORT_CANDIDATE" }
    $selectedRoot = if ($kind -eq "EEG") { $eegRoot.FullName } else { $eegShortRoot.FullName }
    $folderName = Get-SafeName ("{0}__{1}" -f $candidate.Subject, $candidate.Stamp)
    $targetFolder = Join-Path $selectedRoot $folderName
    if (Test-Path -LiteralPath $targetFolder) {
        $targetFolder = "{0}__{1}" -f $targetFolder, $hash.Substring(0, 8)
    }
    New-Item -ItemType Directory -Force -Path $targetFolder | Out-Null
    $copied = Copy-WithUniqueName $candidate.Path $targetFolder
    $record = [pscustomobject][ordered]@{
        kind = $kind
        subject = $candidate.Subject
        timestamp = $candidate.Stamp
        destination_folder = $targetFolder
        destination_files = @($copied)
        source_files = @($candidate.Path)
        duplicate_sources = @()
        sha256 = @($hash)
        bytes = [int64]$candidate.Length
    }
    $seenEegHashes[$hash] = $record
    $inventory.Add($record)
}

$manifest = [ordered]@{
    generated_at = (Get-Date).ToString("o")
    scan_roots = $existingRoots
    acquisition_period = [ordered]@{ start = $periodStart.ToString("yyyy-MM-dd"); end = $periodEnd.ToString("yyyy-MM-dd") }
    rules = @(
        "EEG: TXT has an OpenBCI raw-data header and filename date 2025-03-05 through 2025-03-09.",
        "PPG: non-empty scalar TXT has ch2/ch3, the known ch2 (1)/(2) variant, or the 3.9 participant-name -1/-2 variant.",
        "Identical EEG files and identical PPG pairs are copied once; all discovered duplicate source paths are retained.",
        "Unknown EEG sessions are not assigned to a participant without reliable name evidence."
    )
    unique_ppg_pairs = @($inventory | Where-Object kind -eq "PPG_PAIR").Count
    unique_eeg_sessions = @($inventory | Where-Object kind -eq "EEG").Count
    short_eeg_candidates = @($inventory | Where-Object kind -eq "EEG_SHORT_CANDIDATE").Count
    total_bytes = [int64](($inventory | Measure-Object bytes -Sum).Sum)
    records = @($inventory)
    unpaired = @($unpaired)
}

$manifestPath = Join-Path $destinationFull "manifest.json"
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

$inventoryCsv = foreach ($record in $inventory) {
    [pscustomobject]@{
        kind = $record.kind
        subject = $record.subject
        timestamp = $record.timestamp
        bytes = $record.bytes
        destination_folder = $record.destination_folder
        destination_files = ($record.destination_files -join " | ")
        source_files = ($record.source_files -join " | ")
        duplicate_sources = ($record.duplicate_sources -join " | ")
        sha256 = ($record.sha256 -join " | ")
    }
}
$inventoryCsv | Export-Csv -LiteralPath (Join-Path $destinationFull "inventory.csv") -NoTypeInformation -Encoding UTF8
$unpaired | Export-Csv -LiteralPath (Join-Path $destinationFull "unpaired_ppg.csv") -NoTypeInformation -Encoding UTF8

$ppgCount = @($inventory | Where-Object kind -eq "PPG_PAIR").Count
$eegCount = @($inventory | Where-Object kind -eq "EEG").Count
$shortEegCount = @($inventory | Where-Object kind -eq "EEG_SHORT_CANDIDATE").Count
$unknownEegCount = @($inventory | Where-Object { $_.kind -eq "EEG" -and $_.subject -eq "unknown" }).Count
$totalGiB = [math]::Round($manifest.total_bytes / 1GB, 3)
$readme = @"
# Sensor TXT collection

This directory was generated by code/scripts/data/collect_sensor_texts.ps1 after scanning the computer's available C:, D:, E:, and F: drives.

- Acquisition window: 2025-03-05 through 2025-03-09
- Unique paired PPG sessions: $ppgCount
- Unique OpenBCI EEG sessions: $eegCount
- Short/header-only EEG candidates kept separately: $shortEegCount
- EEG sessions without reliable participant identity: $unknownEegCount
- Copied size: $totalGiB GiB

Each simultaneous PPG pair is stored in one folder. Exact duplicate copies found elsewhere on the computer are not copied again; their paths remain in manifest.json and inventory.csv. EEG recordings without a reliable name are intentionally stored under unknown__... rather than guessed.
"@
$readme | Set-Content -LiteralPath (Join-Path $destinationFull "README.md") -Encoding UTF8

[pscustomobject]@{
    Destination = $destinationFull
    PpgPairs = $ppgCount
    EegSessions = $eegCount
    ShortEegCandidates = $shortEegCount
    UnknownEegSessions = $unknownEegCount
    UnpairedPpgFiles = $unpaired.Count
    CopiedGiB = $totalGiB
    Manifest = $manifestPath
} | Format-List
