"""Data entry point for staging the read-only 927 MP4 drop.

The mapping is intentionally conservative: videos with a clear chronological or
coarse transcript match are attached to existing sessions; otherwise a new
video-only session is created.  The short non-session clip is retained under
``_unresolved_sensor_candidates``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import av

DATASET_NAME = "bci_subjects_ept_v1"
VIDEO_NAME_RE = re.compile(r"^VID_?(?P<date>\d{8})_?(?P<time>\d{6})\.mp4$", re.IGNORECASE)


ASSIGNMENTS: list[dict[str, Any]] = [
    # Existing sessions: identity and/or device chronology provide the pairing.
    {"video": "VID_20250305_211356.mp4", "session": "session_004", "clip": 328.46, "confidence": "high", "basis": "ASR self-introduction says 贺思雨/HSY; same 2025-03-05 acquisition block as the existing HSY DOCX and sensors."},
    {"video": "VID_20250305_220220.mp4", "session": "session_017", "clip": 154.30, "confidence": "high", "basis": "ASR says 赵云露 and the content matches the existing ZYL transcript/video; staged as a high-resolution 927 supplement."},
    {"video": "VID20250306191525.mp4", "session": "session_015", "clip": 120.62, "confidence": "high", "basis": "ASR says 张禾佳; video follows the named ZHJ EEG/PPG acquisition."},
    {"video": "VID20250306194315.mp4", "session": "session_018", "clip": 76.92, "confidence": "high", "basis": "Chronological order and near-overlap with the ZYX PPG start at 19:44:15; coarse transcript matches the session interview."},
    {"video": "VID20250306200329.mp4", "session": "session_006", "clip": 0.0, "confidence": "high", "basis": "Video begins 34 seconds after the named LMX PPG start and follows the March 6 subject order."},
    {"video": "VID20250306201741.mp4", "session": "session_011", "clip": 0.0, "confidence": "high", "basis": "Video starts 72 seconds before the named WYB PPG and is the next March 6 interview."},
    {"video": "VID20250306204031.mp4", "session": "session_012", "clip": 0.0, "confidence": "high", "basis": "Video starts 14 seconds before the named XFX PPG and is the final March 6 interview."},
    {"video": "VID20250308101746.mp4", "session": "session_008", "clip": 0.0, "confidence": "high", "basis": "Video starts 38 seconds after the LZY facial-action stream; date and interview order agree."},
    {"video": "VID20250308102636.mp4", "session": "session_009", "clip": 0.0, "confidence": "high", "basis": "Video follows the WHZ facial-action stream by about 94 seconds; date and interview order agree."},
    {"video": "VID20250308104038.mp4", "session": "session_010", "clip": 0.0, "confidence": "high", "basis": "Third March 8 morning interview, matching the WQ facial-action/DOCX order and coarse transcript."},
    {"video": "VID20250308105228.mp4", "session": "session_002", "clip": 0.0, "confidence": "high", "basis": "Fourth March 8 morning interview, matching the DYN facial-action/DOCX order and coarse transcript."},
    {"video": "VID20250308110310.mp4", "session": "session_003", "clip": 0.0, "confidence": "high", "basis": "ASR says 范志成/FZC; start is 139 seconds after the FZC facial-action stream."},
    # No matching DOCX/session: create stable video-only sessions.
    {"video": "VID_20250308_141718.mp4", "session": "session_030", "subject": "video_20250308_141718", "identity_candidate": "林伟爽", "clip": 41.82, "confidence": "new_session", "basis": "No existing DOCX has this 14:17 acquisition time or interview content."},
    {"video": "VID_20250308_143059.mp4", "session": "session_031", "subject": "video_20250308_143059", "identity_candidate": "韩天乐", "clip": 11.0, "confidence": "new_session", "basis": "No existing DOCX has this 14:30 acquisition time or interview content."},
    {"video": "VID_20250308_144809.mp4", "session": "session_032", "subject": "video_20250308_144809", "identity_candidate": None, "clip": 31.68, "confidence": "new_session", "basis": "Architecture-to-law transfer interview has no matching existing DOCX."},
    {"video": "VID_20250308_151649.mp4", "session": "session_033", "subject": "video_20250308_151649", "identity_candidate": None, "clip": 8.0, "confidence": "new_session", "basis": "Go-club president interview has no matching existing DOCX."},
    {"video": "VID_20250308_201044.mp4", "session": "session_034", "subject": "video_20250308_201044", "identity_candidate": None, "clip": 28.5, "confidence": "new_session", "basis": "Information-management product-manager interview has no matching existing DOCX."},
    {"video": "VID_20250308_202518.mp4", "session": "session_035", "subject": "video_20250308_202518", "identity_candidate": None, "clip": 22.0, "confidence": "new_session", "basis": "Economics graduate-admission interview has no matching existing DOCX."},
    {"video": "VID_20250308_205022.mp4", "session": "session_036", "subject": "video_20250308_205022", "identity_candidate": "赵岳", "clip": 0.0, "confidence": "new_session", "basis": "News/communication self-introduction has no matching existing DOCX."},
    {"video": "VID_20250308_210733.mp4", "session": "session_037", "subject": "video_20250308_210733", "identity_candidate": "陆瑶", "clip": 0.0, "confidence": "new_session", "basis": "Earth-science table-tennis-club interview differs from the existing 李瑶 DOCX and occurs on the prior day."},
    {"video": "VID_20250308_214530.mp4", "session": "session_038", "subject": "video_20250308_214530", "identity_candidate": "黄奥晨", "clip": 0.0, "confidence": "new_session", "basis": "3D hair reconstruction interview has no matching existing DOCX."},
    {"video": "VID_20250309_142903.mp4", "session": "session_039", "subject": "video_20250309_142903", "identity_candidate": "吉世宇", "clip": 96.0, "confidence": "new_session", "basis": "Student-union minister interview does not match the content of existing March 9 DOCX sessions."},
    {"video": "VID_20250309_150935.mp4", "session": "session_040", "subject": "video_20250309_150935", "identity_candidate": "范兴岳", "clip": 0.0, "confidence": "new_session", "basis": "Mathematics-school self-introduction has no matching existing DOCX."},
    {"video": "VID_20250309_153512.mp4", "session": "session_041", "subject": "video_20250309_153512", "identity_candidate": None, "clip": 0.0, "confidence": "new_session", "basis": "Nankai computer-science class-committee interview has no matching existing DOCX."},
]

UNRESOLVED = {
    "video": "VID_20250309_144337.mp4",
    "basis": "Short setup/outtake without a complete self-introduction; retained but not promoted to a subject session.",
}


def read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recorded_at(path: Path) -> datetime:
    match = VIDEO_NAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected video name: {path.name}")
    return datetime.strptime(match.group("date") + match.group("time"), "%Y%m%d%H%M%S")


def media_duration(path: Path) -> float:
    with av.open(str(path), mode="r") as container:
        if container.duration is None:
            raise ValueError(f"Missing media duration: {path}")
        return float(container.duration / av.time_base)


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def encode_video(ffmpeg: Path, source: Path, output: Path, clip_start: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + ".partial" + output.suffix)
    if partial.exists():
        partial.unlink()
    # The phone originals are already H.264/AAC.  Stream-copying preserves their
    # quality and makes staging 18+ GB practical; input seeking trims to the
    # nearest preceding keyframe, which intentionally keeps a small context pad.
    run([
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{clip_start:.3f}", "-i", str(source),
        "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
        "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(partial),
    ])
    os.replace(partial, output)


def extract_audio(ffmpeg: Path, video: Path, output: Path) -> None:
    partial = output.with_name(output.stem + ".partial" + output.suffix)
    if partial.exists():
        partial.unlink()
    run([
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(partial),
    ])
    os.replace(partial, output)


def staged_record(
    *, source: Path, staged: Path, dataset_root: Path, session_id: str,
    subject_id: str, confidence: str, basis: str, clip_start: float,
    source_duration: float, kind: str,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "subject_id": subject_id,
        "source_relative_path": f"927/{source.name}",
        "source_absolute_path": str(source),
        "staged_relative_path": staged.relative_to(dataset_root).as_posix(),
        "bytes": staged.stat().st_size,
        "sha256": sha256(staged),
        "pairing_confidence": confidence,
        "pairing_basis": basis,
        "source_recorded_at": recorded_at(source).isoformat(timespec="seconds"),
        "source_duration_seconds": round(source_duration, 3),
        "clip_start_seconds": round(clip_start, 3),
        "staged_duration_seconds": round(media_duration(staged), 3) if kind == "video" else None,
        "derivation": (
            "trimmed to the nearest preceding keyframe and remuxed without re-encoding"
            if kind == "video"
            else "16 kHz mono PCM audio extracted from the staged aligned video"
        ),
    }


def update_metadata(
    dataset_root: Path, item: dict[str, Any], video_record: dict[str, Any],
    audio_record: dict[str, Any] | None,
) -> dict[str, Any]:
    session_id = item["session"]
    session_root = dataset_root / session_id
    metadata_path = session_root / "session_metadata.json"
    metadata = read_json(metadata_path, {})
    is_new = not bool(metadata)
    subject_id = metadata.get("subject_id") or item["subject"]
    if is_new:
        metadata = {
            "dataset": DATASET_NAME,
            "session_id": session_id,
            "subject_id": subject_id,
            "subject_display_name_candidate": item.get("identity_candidate"),
            "acquisition_date": recorded_at(Path(item["source_path"])).date().isoformat(),
            "annotation_mode": "none",
            "raw_only_requires_reprocessing": True,
            "available_sources": {
                "eeg": False, "paired_ppg": False, "facial_actions": False,
                "openface_features": False, "audio": True, "video": True,
                "annotated_transcript": False,
            },
            "missing_source_reasons": {
                "eeg": "source_artifact_absent_or_not_reliably_paired",
                "paired_ppg": "source_artifact_absent_or_not_reliably_paired",
                "facial_actions": "source_artifact_absent_or_not_reliably_paired",
                "openface_features": "source_artifact_absent_or_not_reliably_paired",
                "annotated_transcript": "no_matching_docx_found",
            },
            "time_pairing": {
                "status": "video_only_new_session",
                "confidence": "high_for_session_boundary_identity_unverified",
                "note": item["basis"],
            },
            "files": [],
        }
    files = [
        record for record in metadata.get("files", [])
        if not (
            record.get("source_relative_path") == f"927/{item['video']}"
            and record.get("staged_relative_path", "").endswith((".mp4", ".wav"))
        )
    ]
    files.append(video_record)
    if audio_record is not None:
        files.append(audio_record)
    metadata["files"] = files
    metadata.setdefault("available_sources", {})["video"] = True
    if audio_record is not None:
        metadata["available_sources"]["audio"] = True
    metadata.setdefault("missing_source_reasons", {}).pop("video", None)
    if audio_record is not None:
        metadata["missing_source_reasons"].pop("audio", None)
    metadata["video_pairing_927"] = {
        "source": item["video"],
        "clip_start_seconds": item["clip"],
        "confidence": item["confidence"],
        "basis": item["basis"],
    }
    write_json(metadata_path, metadata)
    write_json(session_root / "source_manifest.json", metadata)
    return metadata


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve(strict=True)
    dataset_root = args.dataset_root.resolve(strict=True)
    ffmpeg = args.ffmpeg.resolve(strict=True)
    audit_rows: list[dict[str, Any]] = []

    for index, raw_item in enumerate(ASSIGNMENTS, start=1):
        item = dict(raw_item)
        source = (source_root / item["video"]).resolve(strict=True)
        item["source_path"] = str(source)
        session_root = dataset_root / item["session"]
        session_root.mkdir(parents=True, exist_ok=True)
        metadata = read_json(session_root / "session_metadata.json", {})
        subject_id = metadata.get("subject_id") or item["subject"]
        suffix = "_video_927_aligned.mp4" if item["session"] == "session_017" else "_video.mp4"
        video_output = session_root / f"{item['session']}{suffix}"
        audio_output = session_root / f"{item['session']}_audio.wav"
        print(f"[{index}/{len(ASSIGNMENTS)}] {source.name} -> {item['session']}", flush=True)
        if not video_output.is_file():
            encode_video(ffmpeg, source, video_output, float(item["clip"]))
        source_duration = media_duration(source)
        video_record = staged_record(
            source=source, staged=video_output, dataset_root=dataset_root,
            session_id=item["session"], subject_id=subject_id,
            confidence=item["confidence"], basis=item["basis"],
            clip_start=float(item["clip"]), source_duration=source_duration, kind="video",
        )
        audio_record = None
        if not (session_root / f"{item['session']}_audio.mp3").is_file():
            if not audio_output.is_file():
                extract_audio(ffmpeg, video_output, audio_output)
            audio_record = staged_record(
                source=source, staged=audio_output, dataset_root=dataset_root,
                session_id=item["session"], subject_id=subject_id,
                confidence=item["confidence"], basis=item["basis"],
                clip_start=float(item["clip"]), source_duration=source_duration, kind="audio",
            )
        metadata = update_metadata(dataset_root, item, video_record, audio_record)
        start = recorded_at(source)
        audit_rows.append({
            "source_video": source.name,
            "recorded_at": start.isoformat(timespec="seconds"),
            "source_end_at": (start + timedelta(seconds=source_duration)).isoformat(timespec="seconds"),
            "source_duration_seconds": round(source_duration, 3),
            "status": "assigned_existing_session" if int(item["session"].split("_")[1]) <= 29 else "created_new_session",
            "session_id": item["session"],
            "subject_id": metadata["subject_id"],
            "identity_candidate_from_coarse_asr": item.get("identity_candidate"),
            "confidence": item["confidence"],
            "clip_start_seconds": item["clip"],
            "staged_video": str(video_output),
            "staged_audio": str(audio_output) if audio_record else None,
            "basis": item["basis"],
        })

    # Preserve the short non-session clip in the unresolved area.
    unresolved_source = (source_root / UNRESOLVED["video"]).resolve(strict=True)
    unresolved_root = dataset_root / "_unresolved_sensor_candidates" / "video_927"
    unresolved_output = unresolved_root / UNRESOLVED["video"]
    if not unresolved_output.is_file():
        encode_video(ffmpeg, unresolved_source, unresolved_output, 0.0)
    unresolved_duration = media_duration(unresolved_source)
    unresolved_start = recorded_at(unresolved_source)
    audit_rows.append({
        "source_video": unresolved_source.name,
        "recorded_at": unresolved_start.isoformat(timespec="seconds"),
        "source_end_at": (unresolved_start + timedelta(seconds=unresolved_duration)).isoformat(timespec="seconds"),
        "source_duration_seconds": round(unresolved_duration, 3),
        "status": "unresolved_non_session_clip",
        "session_id": "UNRESOLVED",
        "subject_id": None,
        "identity_candidate_from_coarse_asr": None,
        "confidence": "low",
        "clip_start_seconds": 0.0,
        "staged_video": str(unresolved_output),
        "staged_audio": None,
        "basis": UNRESOLVED["basis"],
    })

    write_json(dataset_root / "video_pairing_audit.json", audit_rows)
    write_csv(dataset_root / "video_pairing_audit.csv", audit_rows)

    # Refresh the global source manifest from the session-local manifests.
    root_manifest_path = dataset_root / "source_manifest.json"
    root_manifest = read_json(root_manifest_path, {})
    session_metadata = [
        read_json(path, {})
        for path in sorted(dataset_root.glob("session_*/session_metadata.json"))
    ]
    session_metadata = [item for item in session_metadata if item]
    root_manifest.update({
        "dataset": DATASET_NAME,
        "raw_root": str(dataset_root),
        "sessions": session_metadata,
        "files": [record for metadata in session_metadata for record in metadata.get("files", [])],
        "time_pairing_revision": "2026-09-27_time_pairing_plus_927_video_augmentation",
    })
    write_json(root_manifest_path, root_manifest)

    # Extend the raw session index without changing the frozen processed split.
    index_path = dataset_root / "session_index.json"
    index = read_json(index_path, {"dataset": DATASET_NAME, "sessions": []})
    by_id = {item["session_id"]: item for item in index.get("sessions", [])}
    for item in ASSIGNMENTS:
        if int(item["session"].split("_")[1]) <= 29:
            continue
        by_id[item["session"]] = {
            "session_id": item["session"],
            "subject_id": item["subject"],
            "split": "unassigned",
            "raw_path": f"raw/{DATASET_NAME}/{item['session']}",
            "processed_path": None,
            "status": "raw_video_only_requires_reprocessing",
        }
    index["sessions"] = sorted(by_id.values(), key=lambda row: row["session_id"])
    write_json(index_path, index)

    # Add video rows to the pre-existing time-pairing audit.
    pairing_path = dataset_root / "pairing_audit.json"
    pairing = read_json(pairing_path, [])
    pairing = [row for row in pairing if row.get("modality") != "video_927"]
    pairing.extend({
        "session_id": row["session_id"],
        "subject_id": row["subject_id"],
        "modality": "video_927",
        "status": row["status"],
        "confidence": row["confidence"],
        "source": str(source_root / row["source_video"]),
        "destination": row["staged_video"],
        "evidence": row["basis"],
    } for row in audit_rows)
    write_json(pairing_path, pairing)
    write_csv(dataset_root / "pairing_audit.csv", pairing)

    compliance_path = dataset_root / "session_compliance.json"
    compliance = read_json(compliance_path, {})
    if compliance:
        compliance["stale_after_time_pairing_revision"] = True
        compliance["current_raw_sessions"] = len(session_metadata)
        compliance["stale_reason"] = "Sessions 019-041 and the 927 video/audio additions require preprocessing and a new compliance audit."
        write_json(compliance_path, compliance)

    status = f"""# 927 video pairing status

- Source directory was treated as read-only: `{source_root}`.
- Videos assigned to existing sessions: 12.
- New video-only subject sessions created: 12 (`session_030` through `session_041`).
- Non-session short clip retained as unresolved: 1.
- Total staged session videos from the 927 drop: 24 of 25.
- Aligned 16 kHz mono audio was derived for sessions that did not already have a separate audio file.
- The processed dataset and frozen split were not overwritten; re-run preprocessing before experiments use the new raw sessions/modalities.

See `video_pairing_audit.csv` for the complete source-to-session mapping and clip offsets.
"""
    (dataset_root / "VIDEO_PAIRING_STATUS.md").write_text(status, encoding="utf-8")
    print(f"Completed: {len(audit_rows) - 1} assigned, 1 unresolved", flush=True)


if __name__ == "__main__":
    main()
