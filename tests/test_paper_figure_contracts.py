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


def _write_aggregate(path: Path, *, provenance_sha256: str, seeds: list[int]) -> None:
    payload = {
        "aggregate": {
            "frame.average_precision": {"mean": 0.6, "n": 3},
            "frame.brier_score": {"mean": 0.2, "n": 3},
            "event.event_map": {"mean": 0.4, "n": 3},
        },
        "protocol": {"positive_class": 0},
        "data": {"num_sequences": 2},
        "calibration_protocol": {"selection_split": "validation"},
        "seeds": seeds,
        "provenance_sha256": provenance_sha256,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_modality_aggregates_require_one_source_and_data_provenance(tmp_path: Path) -> None:
    digest = "a" * 64
    paths = []
    for name, seeds in (("full", [13]), ("video", [42])):
        path = tmp_path / f"{name}.json"
        _write_aggregate(path, provenance_sha256=digest, seeds=seeds)
        paths.append(path)

    records, sources, seeds, provenance = FIGURE._records_from_aggregates(
        [
            f"Full={paths[0]}",
            f"Video={paths[1]}",
        ]
    )

    assert len(records) == 6
    assert sources == paths
    assert seeds == {"Full": [13], "Video": [42]}
    assert provenance == {"Full": digest, "Video": digest}


def test_modality_aggregates_reject_provenance_drift(tmp_path: Path) -> None:
    full = tmp_path / "full.json"
    video = tmp_path / "video.json"
    _write_aggregate(full, provenance_sha256="a" * 64, seeds=[13])
    _write_aggregate(video, provenance_sha256="b" * 64, seeds=[42])

    with pytest.raises(ValueError, match="source/data provenance"):
        FIGURE._records_from_aggregates(
            [f"Full={full}", f"Video={video}"]
        )
