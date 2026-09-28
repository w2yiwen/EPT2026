"""Audit label direction and target-speaker consistency for a prepared cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from eptnet.config import load_config
from eptnet.data import StitchedManifestDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def audit(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    positive_class = int(config["evaluation"]["positive_class"])
    if positive_class != 0:
        raise ValueError("This cohort contract requires label 0 to mean deception")

    manifests = {
        split: Path(config["data"][f"{split}_manifest"]).resolve()
        for split in ("train", "val", "test")
    }
    dataset_roots = {path.parent.parent for path in manifests.values()}
    if len(dataset_roots) != 1:
        raise ValueError("All manifests must belong to one prepared dataset")
    dataset_root = next(iter(dataset_roots))
    summary_path = dataset_root / "dataset_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    session_summaries = summary.get("sessions", {})

    records: list[dict[str, Any]] = []
    failures: list[str] = []
    split_counts: dict[str, dict[str, int]] = {}
    for split, manifest in manifests.items():
        dataset = StitchedManifestDataset(str(manifest), require_targets=True)
        counts = {"sessions": 0, "target_valid_steps": 0, "deception_steps": 0}
        for sample in dataset:
            session_id = str(sample["metadata"]["session_id"])
            labels = sample["labels"].long()
            target_mask = sample["target_mask"].bool()
            positive_mask = sample["positive_mask"].bool()
            observed_labels = sorted(int(value) for value in torch.unique(labels).tolist())
            expected_positive = target_mask & labels.eq(positive_class)
            direction_mismatches = int((positive_mask != expected_positive).sum().item())
            if observed_labels != [0, 1]:
                failures.append(f"{session_id}: observed labels are {observed_labels}, not [0, 1]")
            if direction_mismatches:
                failures.append(
                    f"{session_id}: {direction_mismatches} positive_mask/label direction mismatches"
                )

            item = session_summaries.get(session_id, {})
            annotation = item.get("annotation", {})
            alignment = item.get("alignment", {}).get("whisper_alignment", {})
            annotation_speaker = str(annotation.get("target_speaker_id", ""))
            aligned_speaker = str(
                alignment.get("statistics", {}).get(
                    "target_speaker_id", alignment.get("target_speaker_id", "")
                )
            )
            speaker_consistent = (
                bool(annotation_speaker)
                and bool(aligned_speaker)
                and annotation_speaker == aligned_speaker
            )
            if not speaker_consistent:
                failures.append(
                    f"{session_id}: annotation speaker {annotation_speaker!r} "
                    f"disagrees with alignment speaker {aligned_speaker!r}"
                )

            target_valid_steps = int(target_mask.sum().item())
            deception_steps = int(expected_positive.sum().item())
            summary_target_steps = int(item.get("target_valid_steps", -1))
            if summary_target_steps != target_valid_steps:
                failures.append(
                    f"{session_id}: tensor target steps {target_valid_steps} != summary "
                    f"{summary_target_steps}"
                )
            counts["sessions"] += 1
            counts["target_valid_steps"] += target_valid_steps
            counts["deception_steps"] += deception_steps
            records.append(
                {
                    "session_id": session_id,
                    "split": split,
                    "target_speaker_id": annotation_speaker,
                    "alignment_target_speaker_id": aligned_speaker or None,
                    "speaker_consistent": speaker_consistent,
                    "observed_labels": observed_labels,
                    "label_semantics": {"0": "deception", "1": "truth"},
                    "target_valid_steps": target_valid_steps,
                    "deception_steps": deception_steps,
                    "deception_prevalence": deception_steps / target_valid_steps,
                    "positive_mask_direction_mismatches": direction_mismatches,
                    "boundary_positive_entries": int(sample["boundaries"].sum().item()),
                }
            )
        split_counts[split] = counts

    return {
        "schema_version": 1,
        "status": "passed" if not failures else "failed",
        "scope": "prepared tensor structure and recorded target-speaker provenance",
        "dataset": config["data"]["dataset_name"],
        "config": str(config_path.as_posix()),
        "positive_class": positive_class,
        "label_semantics": {"0": "deception", "1": "truth"},
        "checks": {
            "every_session_contains_binary_labels": True,
            "positive_mask_equals_target_mask_and_label_zero": True,
            "target_speaker_matches_alignment_provenance": True,
            "summary_counts_match_tensors": True,
        },
        "split_counts": split_counts,
        "sessions": records,
        "failures": failures,
        "source_sha256": {
            "dataset_summary.json": _sha256(summary_path),
            **{f"{split}_manifest": _sha256(path) for split, path in manifests.items()},
        },
        "limitation": (
            "Passing proves internal direction and speaker-provenance consistency; it cannot "
            "prove that the original human highlights are semantically correct. Independent "
            "data-owner confirmation remains required."
        ),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Label and target-speaker direction audit",
        "",
        f"Status: **{payload['status']}**",
        "",
        "Frozen semantics: `0=deception`, `1=truth`; metrics use class 0 as positive.",
        "",
        "| Session | Split | Target speaker | Valid steps | Deception | Prevalence | Direction mismatches |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["sessions"]:
        lines.append(
            f"| {row['session_id']} | {row['split']} | {row['target_speaker_id']} | "
            f"{row['target_valid_steps']} | {row['deception_steps']} | "
            f"{row['deception_prevalence']:.3f} | "
            f"{row['positive_mask_direction_mismatches']} |"
        )
    lines.extend(["", "## Limitation", "", payload["limitation"], ""])
    if payload["failures"]:
        lines.extend(["## Failures", ""] + [f"- {item}" for item in payload["failures"]])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    args = parser.parse_args()
    payload = audit(args.config)
    _write_json(args.json_output, payload)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(
        _markdown(payload), encoding="utf-8", newline="\n"
    )
    print(json.dumps({"status": payload["status"], "json": str(args.json_output)}))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
