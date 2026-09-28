from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from eptnet import provenance
from eptnet.provenance import (
    build_provenance_fingerprint,
    git_state,
    manifest_fingerprint,
    portable_path,
    portable_python_command,
    required_artifact_fingerprints,
    sha256_file,
    source_tree_fingerprint,
)


def _write_source_tree(root: Path, marker: bytes = b"same") -> None:
    files = {
        "src/eptnet/model.py": b"MODEL:" + marker,
        "configs/default.yaml": b"seed: 42\n" + marker,
        "scripts/experiments/train.sh": b"python -m eptnet.train\n" + marker,
        "tests/test_smoke.py": b"def test_smoke(): assert True\n" + marker,
        "requirements.txt": b"numpy\n" + marker,
        "requirements-lock.txt": b"numpy==1.26.4\n" + marker,
        "pytest.ini": b"[pytest]\ntestpaths = tests\n" + marker,
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _write_dataset(root: Path, tensor_contents: tuple[bytes, ...] = (b"a", b"b")) -> None:
    (root / "manifests").mkdir(parents=True)
    (root / "windows" / "train").mkdir(parents=True)
    for name, content in zip(("a.pt", "b.pt"), tensor_contents, strict=False):
        (root / "windows" / "train" / name).write_bytes(content)
    records = [
        {"sample_id": name[0], "tensor_file": f"../windows/train/{name}"}
        for name in ("a.pt", "b.pt")
    ]
    (root / "manifests" / "train.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    (root / "feature_schema.json").write_text('{"features": ["x"]}\n', encoding="utf-8")
    (root / "dataset_summary.json").write_text('{"samples": 2}\n', encoding="utf-8")
    (root / "normalization_stats.npz").write_bytes(b"npz fixture")


def test_sha256_file_matches_standard_digest_and_missing_fails(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"research artifact")

    assert sha256_file(path) == hashlib.sha256(b"research artifact").hexdigest()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        sha256_file(tmp_path / "missing.bin")


def test_source_tree_is_portable_deterministic_and_allowlisted(tmp_path: Path) -> None:
    first = tmp_path / "machine-a" / "project"
    second = tmp_path / "machine-b" / "project"
    _write_source_tree(first)
    _write_source_tree(second)

    baseline = source_tree_fingerprint(first)
    mirror = source_tree_fingerprint(second)

    assert baseline == mirror
    assert baseline["file_count"] == 7
    assert {record["path"] for record in baseline["files"]} == {
        "src/eptnet/model.py",
        "configs/default.yaml",
        "scripts/experiments/train.sh",
        "tests/test_smoke.py",
        "requirements.txt",
        "requirements-lock.txt",
        "pytest.ini",
    }
    assert str(first) not in json.dumps(baseline)

    for excluded in ("results/run.json", "data/raw.bin", "cache/index.bin"):
        path = first / excluded
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"generated and excluded")
    cache_file = first / "src" / "eptnet" / "__pycache__" / "model.pyc"
    cache_file.parent.mkdir()
    cache_file.write_bytes(b"compiled cache")
    (first / "src" / ".DS_Store").write_bytes(b"operating-system metadata")
    assert source_tree_fingerprint(first)["sha256"] == baseline["sha256"]


@pytest.mark.parametrize(
    "relative_path",
    [
        "src/eptnet/model.py",
        "configs/default.yaml",
        "scripts/experiments/train.sh",
        "tests/test_smoke.py",
        "requirements.txt",
        "requirements-lock.txt",
        "pytest.ini",
    ],
)
def test_source_tree_changes_when_any_required_content_changes(
    tmp_path: Path, relative_path: str
) -> None:
    root = tmp_path / "project"
    _write_source_tree(root)
    before = source_tree_fingerprint(root)["sha256"]

    with (root / relative_path).open("ab") as handle:
        handle.write(b"changed")

    assert source_tree_fingerprint(root)["sha256"] != before


def test_source_tree_missing_required_path_fails_closed(tmp_path: Path) -> None:
    _write_source_tree(tmp_path)
    (tmp_path / "pytest.ini").unlink()

    with pytest.raises(FileNotFoundError, match="pytest.ini"):
        source_tree_fingerprint(tmp_path)


def test_manifest_fingerprint_preserves_tensor_order_and_content(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    manifest = dataset / "manifests" / "train.jsonl"

    initial = manifest_fingerprint(manifest, artifact_root=dataset)
    assert initial["path"] == "manifests/train.jsonl"
    assert [item["path"] for item in initial["tensor_files"]] == [
        "windows/train/a.pt",
        "windows/train/b.pt",
    ]
    assert initial["tensor_count"] == 2

    lines = manifest.read_text(encoding="utf-8").splitlines()
    manifest.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
    reversed_order = manifest_fingerprint(manifest, artifact_root=dataset)
    assert reversed_order["tensors_sha256"] != initial["tensors_sha256"]
    assert reversed_order["manifest_and_tensors_sha256"] != initial["manifest_and_tensors_sha256"]

    stable_manifest_hash = reversed_order["manifest_sha256"]
    (dataset / "windows" / "train" / "a.pt").write_bytes(b"changed tensor")
    changed_tensor = manifest_fingerprint(manifest, artifact_root=dataset)
    assert changed_tensor["manifest_sha256"] == stable_manifest_hash
    assert (
        changed_tensor["manifest_and_tensors_sha256"]
        != reversed_order["manifest_and_tensors_sha256"]
    )


def test_manifest_missing_tensor_and_escape_fail_closed(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    manifest = dataset / "manifests" / "train.jsonl"
    (dataset / "windows" / "train" / "b.pt").unlink()

    with pytest.raises(FileNotFoundError, match="b.pt"):
        manifest_fingerprint(manifest, artifact_root=dataset)

    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"private")
    manifest.write_text(
        json.dumps({"sample_id": "escape", "tensor_file": "../../outside.pt"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes the permitted root"):
        manifest_fingerprint(manifest, artifact_root=dataset)

    outside_manifest = tmp_path / "outside.jsonl"
    outside_manifest.write_text(
        json.dumps({"sample_id": "a", "tensor_file": "dataset/windows/train/a.pt"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes the permitted root"):
        manifest_fingerprint(outside_manifest, artifact_root=dataset)


@pytest.mark.parametrize(
    "record",
    [
        {"sample_id": "missing"},
        {"sample_id": "absolute", "tensor_file": "/private/a.pt"},
        {"sample_id": "windows", "tensor_file": r"..\windows\a.pt"},
    ],
)
def test_manifest_rejects_nonportable_or_incomplete_records(
    tmp_path: Path, record: dict[str, str]
) -> None:
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    manifest = dataset / "manifests" / "train.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        manifest_fingerprint(manifest, artifact_root=dataset)


def test_required_artifact_hashes_are_named_and_fail_closed(tmp_path: Path) -> None:
    project = tmp_path / "project"
    dataset = project / "data" / "processed" / "fixture"
    _write_dataset(dataset)
    lock = project / "requirements-lock.txt"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_bytes(b"numpy==1.26.4\n")

    records = required_artifact_fingerprints(dataset, lock, path_root=project)
    assert set(records) == {
        "feature_schema",
        "normalization_stats",
        "dataset_summary",
        "requirements_lock",
    }
    assert records["requirements_lock"]["path"] == "requirements-lock.txt"
    assert records["feature_schema"]["path"] == ("data/processed/fixture/feature_schema.json")

    (dataset / "normalization_stats.npz").unlink()
    with pytest.raises(FileNotFoundError, match="normalization_stats.npz"):
        required_artifact_fingerprints(dataset, lock, path_root=project)


def test_complete_provenance_is_deterministic_and_git_optional(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for project in (first, second):
        _write_source_tree(project)
        _write_dataset(project / "data" / "processed" / "fixture")

    first_record = build_provenance_fingerprint(
        first,
        first / "data" / "processed" / "fixture",
        include_git=False,
    )
    second_record = build_provenance_fingerprint(
        second,
        second / "data" / "processed" / "fixture",
        include_git=False,
    )

    assert first_record == second_record
    assert first_record["git"] is None
    assert len(first_record["provenance_sha256"]) == 64
    assert first_record["source_tree"]["sha256"]
    assert first_record["manifests"][0]["manifest_and_tensors_sha256"]
    serialized = json.dumps(first_record)
    assert str(first) not in serialized
    assert str(second) not in serialized


def test_complete_provenance_changes_for_dataset_content(tmp_path: Path) -> None:
    project = tmp_path / "project"
    dataset = project / "data" / "processed" / "fixture"
    _write_source_tree(project)
    _write_dataset(dataset)
    before = build_provenance_fingerprint(project, dataset, include_git=False)

    (dataset / "feature_schema.json").write_text('{"features": ["y"]}\n', encoding="utf-8")
    after = build_provenance_fingerprint(project, dataset, include_git=False)

    assert after["source_tree"]["sha256"] == before["source_tree"]["sha256"]
    assert after["provenance_sha256"] != before["provenance_sha256"]


def test_git_state_gracefully_falls_back_when_git_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing_git(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(provenance.subprocess, "run", missing_git)
    assert git_state(tmp_path) is None


def test_git_state_rejects_an_ancestor_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "standalone-artifact"
    project.mkdir()
    calls: list[tuple[str, ...]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        arguments = tuple(command[1:])
        calls.append(arguments)
        assert arguments == ("rev-parse", "--show-toplevel")
        return subprocess.CompletedProcess(command, 0, f"{tmp_path}\n".encode(), b"")

    monkeypatch.setattr(provenance.subprocess, "run", fake_run)

    assert git_state(project) is None
    assert calls == [("rev-parse", "--show-toplevel")]


def test_git_state_accepts_only_the_exact_project_repository_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        arguments = tuple(command[1:])
        if arguments == ("rev-parse", "--show-toplevel"):
            return subprocess.CompletedProcess(command, 0, f"{tmp_path}\n".encode(), b"")
        if arguments == ("rev-parse", "--verify", "HEAD^{commit}"):
            return subprocess.CompletedProcess(command, 0, f"{commit}\n".encode(), b"")
        if arguments == (
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
            "--",
            ".",
        ):
            return subprocess.CompletedProcess(command, 0, b"", b"")
        raise AssertionError(f"Unexpected Git invocation: {arguments}")

    monkeypatch.setattr(provenance.subprocess, "run", fake_run)

    assert git_state(tmp_path) == {"commit": commit, "dirty": False}


def test_portable_path_preserves_in_tree_identity_without_absolute_parent(
    tmp_path: Path,
) -> None:
    result = tmp_path / "results" / "seed_42" / "test_metrics.json"
    result.parent.mkdir(parents=True)
    result.touch()

    serialized = portable_path(result, root=tmp_path)

    assert serialized == "results/seed_42/test_metrics.json"
    assert str(tmp_path) not in serialized


def test_portable_path_redacts_external_parent(tmp_path: Path) -> None:
    root = tmp_path / "project"
    external = tmp_path / "private-user-name" / "best.pt"
    root.mkdir()

    serialized = portable_path(external, root=root)

    assert serialized == "<external>/best.pt"
    assert "private-user-name" not in serialized


def test_portable_python_command_sanitizes_absolute_arguments(tmp_path: Path) -> None:
    root = tmp_path / "project"
    config = root / "configs" / "default.yaml"
    checkpoint = tmp_path / "private-user-name" / "last.pt"
    config.parent.mkdir(parents=True)

    command = portable_python_command(
        tmp_path / "private-user-name" / "python.exe",
        "eptnet.train",
        ("--config", str(config), "--resume", str(checkpoint)),
        root=root,
    )

    assert command == [
        "python.exe",
        "-m",
        "eptnet.train",
        "--config",
        "configs/default.yaml",
        "--resume",
        "<external>/last.pt",
    ]
    assert "private-user-name" not in " ".join(command)
