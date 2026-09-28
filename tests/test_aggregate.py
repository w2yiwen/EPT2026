from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import eptnet.aggregate as aggregate_module
from eptnet.aggregate import aggregate_result_files


def _payload(
    *,
    seed: int,
    experiment: str = "eptnet_default",
    macro_f1: float = 0.5,
    frame_threshold: float = 0.5,
    positive_class: int = 0,
    latency: float | None = 0.2,
    calibration_enabled: bool = True,
    dense_event_proposals: int = 153,
    fingerprint: str = "a" * 64,
    provenance_fingerprint: str = "c" * 64,
) -> dict:
    return {
        "protocol": {
            "evaluation_unit": "unique chronological rows reconstructed per session",
            "positive_class": positive_class,
            "frame_threshold": frame_threshold,
            "boundary_threshold": 0.5,
            "boundary_tolerance_steps": 1,
        },
        "data": {
            "num_sequences": 1,
            "num_unique_steps": 153,
            "num_target_events": 30,
            "num_predicted_events": seed + 1,
            "num_dense_event_proposals": dense_event_proposals,
        },
        "frame": {"macro_f1": macro_f1, "num_frames": 153},
        "event": {"event_map": 0.4 + seed * 0.01},
        "latency": {"mean_detection_delay_seconds": latency},
        "calibration": (
            {
                "enabled": True,
                "method": "validation_grid_macro_f1",
                "selection_split": "validation",
                "calibration_min_threshold": 0.1,
                "calibration_max_threshold": 0.9,
                "calibration_steps": 81,
                "frame_threshold": frame_threshold,
                "validation_macro_f1": 0.7,
            }
            if calibration_enabled
            else {
                "enabled": False,
                "method": "fixed_threshold",
                "selection_split": None,
                "calibration_min_threshold": None,
                "calibration_max_threshold": None,
                "calibration_steps": None,
                "frame_threshold": frame_threshold,
            }
        ),
        "checkpoint": {
            "seed": seed,
            "experiment": experiment,
            "epoch": 12,
            "path": f"seed_{seed}/best.pt",
            "training_config_sha256": fingerprint,
            "provenance_sha256": provenance_fingerprint,
        },
        "provenance": {
            "schema_version": 1,
            "provenance_sha256": provenance_fingerprint,
        },
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_aggregate_validates_identity_and_reports_partial_metrics(tmp_path: Path):
    first = _write(
        tmp_path / "seed_1.json",
        _payload(seed=1, macro_f1=0.6, frame_threshold=0.42, latency=None),
    )
    second = _write(
        tmp_path / "seed_2.json",
        _payload(seed=2, macro_f1=0.8, frame_threshold=0.57, latency=0.3),
    )

    summary = aggregate_result_files([first, second])

    assert summary["experiment"] == "eptnet_default"
    assert summary["seeds"] == [1, 2]
    assert summary["provenance_sha256"] == "c" * 64
    assert summary["aggregate"]["frame.macro_f1"]["mean"] == pytest.approx(0.7)
    assert summary["aggregate"]["frame.macro_f1"]["n"] == 2
    assert summary["aggregate"]["latency.mean_detection_delay_seconds"]["n"] == 1
    assert summary["aggregate"]["latency.mean_detection_delay_seconds"]["missing"] == 1
    assert summary["incomplete_metrics"] == ["latency.mean_detection_delay_seconds"]
    assert "checkpoint.seed" not in summary["aggregate"]
    assert "protocol.frame_threshold" not in summary["aggregate"]


def test_aggregate_rejects_duplicate_seed(tmp_path: Path):
    first = _write(tmp_path / "first.json", _payload(seed=7))
    second = _write(tmp_path / "second.json", _payload(seed=7, macro_f1=0.9))

    with pytest.raises(ValueError, match="Duplicate evaluation seeds"):
        aggregate_result_files([first, second])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("experiment", "baseline", "different experiments"),
        ("positive_class", 1, "protocol/data mismatch"),
        ("calibration_enabled", False, "protocol/data mismatch"),
        ("dense_event_proposals", 152, "protocol/data mismatch"),
        ("fingerprint", "b" * 64, "different configurations"),
        (
            "provenance_fingerprint",
            "d" * 64,
            "different source or prepared-data provenance",
        ),
    ],
)
def test_aggregate_rejects_incomparable_runs(
    tmp_path: Path, field: str, value: object, message: str
):
    first = _write(tmp_path / "seed_1.json", _payload(seed=1))
    overrides = {field: value}
    second = _write(tmp_path / "seed_2.json", _payload(seed=2, **overrides))

    with pytest.raises(ValueError, match=message):
        aggregate_result_files([first, second])


def test_aggregate_rejects_missing_or_internally_inconsistent_provenance(tmp_path: Path):
    missing_payload = _payload(seed=1)
    missing_payload.pop("provenance")
    missing = _write(tmp_path / "missing.json", missing_payload)
    with pytest.raises(ValueError, match="missing provenance metadata"):
        aggregate_result_files([missing])

    inconsistent_payload = _payload(seed=2)
    inconsistent_payload["checkpoint"]["provenance_sha256"] = "e" * 64
    inconsistent = _write(tmp_path / "inconsistent.json", inconsistent_payload)
    with pytest.raises(ValueError, match="checkpoint and evaluation provenance disagree"):
        aggregate_result_files([inconsistent])


def test_aggregate_cli_writes_json_and_csv_with_lf_only(tmp_path: Path, monkeypatch):
    first = _write(tmp_path / "seed_1.json", _payload(seed=1))
    second = _write(tmp_path / "seed_2.json", _payload(seed=2))
    output = tmp_path / "aggregate.json"
    csv_output = tmp_path / "aggregate.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eptnet.aggregate",
            str(first),
            str(second),
            "--output",
            str(output),
            "--csv",
            str(csv_output),
        ],
    )

    aggregate_module.main()

    for path in (output, csv_output):
        payload = path.read_bytes()
        assert b"\n" in payload
        assert b"\r" not in payload
