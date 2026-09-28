"""Remap legacy normalized MARLIN embeddings onto the aligned audio timeline."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from eptnet.data.session_package import atomic_write_json
from eptnet.data.whisper_alignment import load_alignment, sha256_file

EXCLUDED_SESSION = "session_011"
DIRECT_VERSION = "per_sampled_frame_face_detection_real_time_v2"
REMAP_VERSION = "legacy_embedding_time_remap_v1"
OLD_VERSION = "per_sampled_frame_face_detection_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alignment-dir",
        default="data/cache/bci_subjects_ept_v6_marlin4060_complete12/whisper_alignment",
    )
    parser.add_argument(
        "--legacy-cache",
        default="data/cache/bci_subjects_ept_v6_marlin4060_complete12/marlin",
    )
    parser.add_argument(
        "--output-cache",
        default="data/cache/bci_subjects_ept_v6_marlin4060_aligned11/marlin",
    )
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    alignment_dir = Path(args.alignment_dir).expanduser().resolve()
    legacy_cache = Path(args.legacy_cache).expanduser().resolve()
    output_cache = Path(args.output_cache).expanduser().resolve()
    output_cache.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []
    artifacts = sorted(alignment_dir.glob("session_*.json"))
    selected = [path for path in artifacts if path.stem != EXCLUDED_SESSION]
    if len(selected) != 11:
        raise ValueError(f"Expected 11 aligned artifacts excluding session_011, found {len(selected)}")

    for artifact_path in selected:
        artifact = load_alignment(artifact_path)
        session_id = str(artifact["session_id"])
        video = Path(str(artifact["video_path"]))
        if not video.is_file() or sha256_file(video) != artifact["video_sha256"]:
            raise ValueError(f"Video fingerprint mismatch for {session_id}: {video}")
        digest = str(artifact["video_sha256"])[:16]
        pattern = f"{video.stem}.{digest}.steps*.crop1.{OLD_VERSION}.npz"
        candidates = sorted(legacy_cache.glob(pattern))
        if len(candidates) != 1:
            raise ValueError(f"Expected one legacy MARLIN cache for {session_id}, found {candidates}")
        legacy_path = candidates[0]
        match = re.search(r"\.steps(\d+)\.", legacy_path.name)
        if match is None:
            raise ValueError(f"Could not read legacy step count: {legacy_path}")
        old_steps = int(match.group(1))
        with np.load(legacy_path, allow_pickle=False) as cached:
            old_features = np.asarray(cached["features"], dtype=np.float32)
            old_mask = np.asarray(cached["mask"], dtype=bool)
        if old_features.shape != (old_steps, 384) or old_mask.shape != (old_steps,):
            raise ValueError(f"Invalid legacy MARLIN cache shape: {legacy_path}")

        new_steps = int(artifact["num_steps"])
        step_seconds = float(artifact["step_seconds"])
        offset = float(artifact["video_alignment"]["offset_seconds"])
        video_duration = float(
            artifact["video_alignment"]["video_audio_duration_seconds"]
        )
        target_centers = (
            (np.arange(new_steps, dtype=np.float64) + 0.5) * step_seconds + offset
        )
        legacy_centers = (
            (np.arange(old_steps, dtype=np.float64) + 0.5) * video_duration / old_steps
        )
        source_indices = np.rint(
            target_centers / video_duration * old_steps - 0.5
        ).astype(np.int64)
        valid = (source_indices >= 0) & (source_indices < old_steps)
        clipped = np.clip(source_indices, 0, old_steps - 1)
        valid &= old_mask[clipped]
        features = np.zeros((new_steps, 384), dtype=np.float32)
        features[valid] = old_features[clipped[valid]]
        mask = valid.astype(bool)
        errors = np.abs(target_centers[valid] - legacy_centers[clipped[valid]])

        output_name = (
            f"{video.stem}.{digest}.steps{new_steps}.step{step_seconds}."
            f"offset{offset:.6f}.crop1.{REMAP_VERSION}.npz"
        )
        output_path = output_cache / output_name
        if output_path.exists() and not args.overwrite:
            print(json.dumps({"event": "remap_cache_hit", "session_id": session_id}), flush=True)
        else:
            temporary = output_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                features=features,
                mask=mask,
                source_indices=source_indices,
                alignment_method=np.asarray(REMAP_VERSION),
            )
            temporary.replace(output_path)
            print(json.dumps({"event": "remap_written", "session_id": session_id}), flush=True)
        reports.append(
            {
                "session_id": session_id,
                "legacy_cache": str(legacy_path),
                "remapped_cache": str(output_path),
                "legacy_steps": old_steps,
                "aligned_steps": new_steps,
                "valid_steps": int(mask.sum()),
                "mean_center_error_seconds": float(errors.mean()) if len(errors) else None,
                "maximum_center_error_seconds": float(errors.max()) if len(errors) else None,
                "method": REMAP_VERSION,
                "direct_frame_reextraction": False,
            }
        )
    atomic_write_json(
        output_cache / "remap_report.json",
        {
            "schema_version": 1,
            "method": REMAP_VERSION,
            "excluded_sessions": [EXCLUDED_SESSION],
            "direct_frame_reextraction": False,
            "sessions": reports,
        },
    )
    print(json.dumps({"event": "remap_complete", "sessions": len(reports)}), flush=True)


if __name__ == "__main__":
    main()
