"""Create fail-closed Whisper/reference/video alignment artifacts for the 12-session cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eptnet.data.prepare_bci_subjects import (
    _document_paragraphs,
    _document_text_paragraphs,
    _parse_speaker_turns,
    discover_sessions,
)
from eptnet.data.session_package import atomic_write_json
from eptnet.data.whisper_alignment import (
    ALIGNMENT_SCHEMA_VERSION,
    build_reference_timeline,
    estimate_video_audio_offset,
    load_video_audio,
    sha256_file,
)
from eptnet.models.behavior.audio.wavlm import _load_audio_mono

EXPECTED_SESSIONS = (
    "session_002",
    "session_003",
    "session_004",
    "session_006",
    "session_008",
    "session_009",
    "session_010",
    "session_011",
    "session_012",
    "session_015",
    "session_017",
    "session_018",
)


def _complete_and_current(
    path: Path,
    fingerprints: dict[str, str],
    generation_config: dict[str, object],
) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    stored_config = value.get("generation_config")
    if not isinstance(stored_config, dict):
        return False
    requested_without_coverage = {
        key: item
        for key, item in generation_config.items()
        if key != "minimum_reference_coverage"
    }
    stored_without_coverage = {
        key: item
        for key, item in stored_config.items()
        if key != "minimum_reference_coverage"
    }
    # An artifact accepted under a stricter coverage threshold remains valid
    # when resuming with a lower threshold. This avoids recomputing completed
    # sessions while preserving the exact model/source/config fingerprint.
    coverage_compatible = float(
        stored_config.get("minimum_reference_coverage", -1.0)
    ) >= float(generation_config["minimum_reference_coverage"])
    return (
        value.get("schema_version") == ALIGNMENT_SCHEMA_VERSION
        and value.get("status") == "complete"
        and stored_without_coverage == requested_without_coverage
        and coverage_compatible
        and all(value.get(key) == digest for key, digest in fingerprints.items())
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/raw/bci_subjects_ept_v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="turbo")
    parser.add_argument("--model-cache", default="data/cache/whisper")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--minimum-reference-coverage", type=float, default=0.50)
    parser.add_argument("--minimum-audio-correlation", type=float, default=0.20)
    parser.add_argument("--maximum-offset-seconds", type=float, default=30.0)
    parser.add_argument(
        "--exclude-session",
        action="append",
        default=[],
        help="Session ID to exclude explicitly; may be supplied more than once",
    )
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    import torch
    import whisper

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {args.device}")
    source_root = Path(args.source).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    by_id = {session.session_id: session for session in discover_sessions(source_root)}
    excluded = set(args.exclude_session)
    unknown_exclusions = excluded.difference(EXPECTED_SESSIONS)
    if unknown_exclusions:
        raise ValueError(f"Unknown session exclusions: {sorted(unknown_exclusions)}")
    selected_sessions = tuple(
        session_id for session_id in EXPECTED_SESSIONS if session_id not in excluded
    )
    if len(selected_sessions) < 3:
        raise ValueError("Session exclusions leave fewer than three sessions")
    missing = [session_id for session_id in selected_sessions if session_id not in by_id]
    if missing:
        raise ValueError(f"The fixed 12-session cohort is incomplete: {missing}")

    pending = []
    generation_config = {
        "whisper_model": args.model,
        "language": args.language,
        "step_seconds": 1.0,
        "minimum_reference_coverage": args.minimum_reference_coverage,
        "minimum_audio_correlation": args.minimum_audio_correlation,
        "maximum_offset_seconds": args.maximum_offset_seconds,
        "video_audio_envelope_hz": 50,
    }
    for session_id in selected_sessions:
        session = by_id[session_id]
        if session.annotation is None or session.audio_file is None or session.video_file is None:
            raise ValueError(f"{session_id} lacks annotation/audio/video")
        fingerprints = {
            "annotation_sha256": sha256_file(session.annotation),
            "audio_sha256": sha256_file(session.audio_file),
            "video_sha256": sha256_file(session.video_file),
        }
        target = output_root / f"{session_id}.json"
        if not args.overwrite and _complete_and_current(
            target, fingerprints, generation_config
        ):
            print(json.dumps({"event": "alignment_cache_hit", "session_id": session_id}), flush=True)
            continue
        pending.append((session, fingerprints, target))

    if not pending:
        print(
            json.dumps(
                {
                    "event": "alignment_complete",
                    "sessions": len(selected_sessions),
                    "excluded_sessions": sorted(excluded),
                }
            ),
            flush=True,
        )
        return
    model = whisper.load_model(
        args.model,
        device=args.device,
        download_root=str(Path(args.model_cache).expanduser().resolve()),
    )
    for index, (session, fingerprints, target) in enumerate(pending, start=1):
        print(
            json.dumps(
                {
                    "event": "alignment_start",
                    "session_id": session.session_id,
                    "index": index,
                    "pending_total": len(pending),
                }
            ),
            flush=True,
        )
        audio_16k = _load_audio_mono(session.audio_file, 16_000)
        duration = len(audio_16k) / 16_000
        result = model.transcribe(
            audio_16k,
            language=args.language,
            task="transcribe",
            word_timestamps=True,
            fp16=args.device.startswith("cuda"),
            temperature=0.0,
            condition_on_previous_text=True,
            verbose=False,
        )
        turns = _parse_speaker_turns(
            _document_text_paragraphs(session.annotation), session.annotation
        )
        annotated_turns = _parse_speaker_turns(
            _document_paragraphs(session.annotation),
            session.annotation,
            session.annotation_mode,
        )
        artifact = build_reference_timeline(
            turns,
            annotated_turns,
            result,
            audio_duration_seconds=duration,
            minimum_reference_coverage=args.minimum_reference_coverage,
        )

        external_8k = _load_audio_mono(session.audio_file, 8_000)
        video_8k = load_video_audio(session.video_file, 8_000)
        if not len(video_8k):
            raise ValueError(f"{session.session_id} video has no decodable audio track")
        offset, correlation = estimate_video_audio_offset(
            external_8k,
            video_8k,
            sample_rate=8_000,
            maximum_offset_seconds=args.maximum_offset_seconds,
        )
        if correlation < args.minimum_audio_correlation:
            raise ValueError(
                f"{session.session_id} video/audio correlation {correlation:.3f} is below "
                f"{args.minimum_audio_correlation:.3f}"
            )
        artifact.update(
            {
                "session_id": session.session_id,
                **fingerprints,
                "annotation_path": str(session.annotation),
                "audio_path": str(session.audio_file),
                "video_path": str(session.video_file),
                "whisper_model": args.model,
                "whisper_language": args.language,
                "whisper_version": getattr(whisper, "__version__", "unknown"),
                "generation_config": generation_config,
                "video_alignment": {
                    "method": "embedded_video_audio_vs_external_audio_rms_cross_correlation",
                    "offset_semantics": "video_time = external_audio_time + offset_seconds",
                    "offset_seconds": offset,
                    "correlation": correlation,
                    "external_audio_duration_seconds": len(external_8k) / 8_000,
                    "video_audio_duration_seconds": len(video_8k) / 8_000,
                },
            }
        )
        atomic_write_json(target, artifact)
        print(
            json.dumps(
                {
                    "event": "alignment_complete",
                    "session_id": session.session_id,
                    "reference_coverage": artifact["statistics"][
                        "reference_character_coverage"
                    ],
                    "video_offset_seconds": offset,
                    "video_audio_correlation": correlation,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
