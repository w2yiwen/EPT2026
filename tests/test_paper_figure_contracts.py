from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "fig"
    / "fig05_modality_evidence"
    / "plot_fig05_modality_evidence.py"
)
SPEC = importlib.util.spec_from_file_location("plot_fig05_modality_evidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
FIGURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIGURE)


def _write_aggregate(path: Path, *, provenance_sha256: str) -> None:
    payload = {
        "aggregate": {
            "frame.average_precision": {"mean": 0.6, "n": 3},
            "frame.brier_score": {"mean": 0.2, "n": 3},
            "event.event_map": {"mean": 0.4, "n": 3},
        },
        "protocol": {"positive_class": 0},
        "data": {"num_sequences": 2},
        "calibration_protocol": {"selection_split": "validation"},
        "seeds": [13, 42, 73],
        "provenance_sha256": provenance_sha256,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_modality_aggregates_require_one_source_and_data_provenance(tmp_path: Path) -> None:
    digest = "a" * 64
    paths = []
    for name in ("full", "physiology", "video"):
        path = tmp_path / f"{name}.json"
        _write_aggregate(path, provenance_sha256=digest)
        paths.append(path)

    records, sources = FIGURE._records_from_aggregates(
        [
            f"Full={paths[0]}",
            f"EEG+PPG={paths[1]}",
            f"Video={paths[2]}",
        ]
    )

    assert len(records) == 9
    assert sources == paths


def test_modality_aggregates_reject_provenance_drift(tmp_path: Path) -> None:
    full = tmp_path / "full.json"
    physiology = tmp_path / "physiology.json"
    video = tmp_path / "video.json"
    _write_aggregate(full, provenance_sha256="a" * 64)
    _write_aggregate(physiology, provenance_sha256="b" * 64)
    _write_aggregate(video, provenance_sha256="a" * 64)

    with pytest.raises(ValueError, match="source/data provenance"):
        FIGURE._records_from_aggregates(
            [f"Full={full}", f"EEG+PPG={physiology}", f"Video={video}"]
        )
