from __future__ import annotations

import ast
from pathlib import Path

import torch

import eptnet.engine as engine_module
import eptnet.train as train_module
from eptnet.config import save_resolved_config
from eptnet.data import prepare_bci_subjects
from eptnet.evaluate import _write_events, _write_predictions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_WRITER_SOURCES = (
    PROJECT_ROOT / "src/eptnet/aggregate.py",
    PROJECT_ROOT / "src/eptnet/config.py",
    PROJECT_ROOT / "src/eptnet/engine.py",
    PROJECT_ROOT / "src/eptnet/evaluate.py",
    PROJECT_ROOT / "src/eptnet/train.py",
    PROJECT_ROOT / "src/eptnet/data/audit.py",
    PROJECT_ROOT / "src/eptnet/data/prepare_bci.py",
    PROJECT_ROOT / "src/eptnet/data/prepare_bci_subjects.py",
    PROJECT_ROOT / "scripts/audit/audit_bci_subjects.py",
    PROJECT_ROOT / "scripts/audit/audit_parameter_fairness.py",
    PROJECT_ROOT / "scripts/reporting/build_release_report.py",
    PROJECT_ROOT / "scripts/reporting/gen_fig_experiments.py",
)


def _keyword_literal(call: ast.Call, name: str) -> object | None:
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    return None


def _assert_lf_only(path: Path) -> None:
    payload = path.read_bytes()
    assert b"\n" in payload
    assert b"\r" not in payload


def test_published_text_writers_declare_deterministic_lf() -> None:
    """Keep future release writers from silently reintroducing platform CRLF."""
    violations: list[str] = []
    for path in PUBLISHED_WRITER_SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            function_name = (
                function.attr
                if isinstance(function, ast.Attribute)
                else function.id
                if isinstance(function, ast.Name)
                else None
            )
            if function_name == "write_text" and _keyword_literal(node, "newline") != "\n":
                violations.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: write_text")
            if function_name in {"open", "fdopen"}:
                mode_index = 1 if function_name == "fdopen" or isinstance(function, ast.Name) else 0
                mode = _keyword_literal(node, "mode")
                if mode is None and len(node.args) > mode_index:
                    positional_mode = node.args[mode_index]
                    if isinstance(positional_mode, ast.Constant):
                        mode = positional_mode.value
                if mode == "w" and _keyword_literal(node, "newline") not in {"\n", ""}:
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: {function_name}"
                    )
            if function_name in {"writer", "DictWriter"}:
                if _keyword_literal(node, "lineterminator") != "\n":
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: {function_name}"
                    )
    assert not violations, "Non-deterministic text writers:\n" + "\n".join(violations)


def test_core_json_yaml_and_manifest_writers_emit_lf_only(tmp_path: Path) -> None:
    outputs = {
        "engine": tmp_path / "engine.json",
        "train": tmp_path / "train.json",
        "config": tmp_path / "resolved.yaml",
        "subject_summary": tmp_path / "subject_summary.json",
    }
    payload = {"alpha": [1, 2], "unicode": "可复现"}

    engine_module._atomic_write_json(payload, outputs["engine"])
    train_module._atomic_write_json(payload, outputs["train"])
    save_resolved_config(payload, str(outputs["config"]))
    prepare_bci_subjects._write_json(outputs["subject_summary"], payload)

    for path in outputs.values():
        _assert_lf_only(path)


def test_evaluation_artifact_writers_emit_lf_only(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    events_path = tmp_path / "events.json"
    sequence = {
        "sample_id": "session_001",
        "row_indices": torch.tensor([0, 1]),
        "class_logits": torch.tensor([[2.0, -1.0], [-1.0, 2.0]]),
        "boundary_logits": torch.zeros(2, 2),
        "offsets": torch.zeros(2, 2),
        "labels": torch.tensor([0, 1]),
        "target_mask": torch.tensor([True, True]),
    }

    _write_predictions(
        predictions_path,
        [sequence],
        positive_class=0,
        frame_threshold=0.5,
    )
    _write_events(
        events_path,
        [[{"start": 0.0, "end": 0.0, "score": 0.9, "emit_step": 0.0}]],
        [[{"start": 0.0, "end": 0.0}]],
    )

    _assert_lf_only(predictions_path)
    _assert_lf_only(events_path)
