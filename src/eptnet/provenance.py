from __future__ import annotations

import hashlib
import json
import os
import subprocess
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

PROVENANCE_SCHEMA_VERSION = 1

# These paths define the executable research artifact. Keeping the default an
# allowlist means generated data, results, figures, logs, and notebooks cannot
# silently perturb the source identity.
DEFAULT_SOURCE_TREE_PATHS: tuple[str, ...] = (
    "src",
    "configs",
    "scripts",
    "requirements.txt",
    "requirements-lock.txt",
)
OPTIONAL_SOURCE_TREE_PATHS: tuple[str, ...] = (
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "tox.ini",
)
REQUIRED_ARTIFACT_FILENAMES: Mapping[str, str] = {
    "feature_schema": "feature_schema.json",
    "normalization_stats": "normalization_stats.npz",
    "dataset_summary": "dataset_summary.json",
}

_HASH_CHUNK_BYTES = 1024 * 1024
_EXCLUDED_SOURCE_DIRECTORIES = frozenset(
    {
        "__pycache__",
        ".cache",
        "cache",
        "caches",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
    }
)
_EXCLUDED_SOURCE_SUFFIXES = frozenset({".pyc", ".pyo"})
_EXCLUDED_SOURCE_FILENAMES = frozenset({".DS_Store", "Thumbs.db"})


def training_config_fingerprint(config: Mapping[str, Any]) -> str:
    """Hash training-relevant settings while ignoring run placement and device.

    Seed, output directory, and execution device identify a concrete run rather
    than the model/training comparison.  Keeping this helper in the lightweight
    provenance module lets evaluation and read-only artifact verification share
    one canonical implementation.
    """

    experiment = dict(config.get("experiment", {}))
    experiment.pop("seed", None)
    experiment.pop("output_dir", None)
    training = dict(config.get("training", {}))
    training.pop("device", None)
    canonical = {
        "experiment": experiment,
        "data": config.get("data", {}),
        "model": config.get("model", {}),
        "loss": config.get("loss", {}),
        "training": training,
    }
    return _canonical_sha256(canonical)


def portable_path(value: str | Path, root: str | Path | None = None) -> str:
    """Return a path identity without serializing machine-local parents.

    Paths below ``root`` (the working directory by default) use POSIX
    separators. External paths retain only their basename so metadata cannot
    disclose a user name or home directory.
    """

    base = Path.cwd().resolve() if root is None else Path(root).resolve()
    resolved = Path(value).expanduser().resolve()
    try:
        relative = resolved.relative_to(base)
    except ValueError:
        name = resolved.name or "redacted"
        return f"<external>/{name}"
    return "." if not relative.parts else relative.as_posix()


def portable_python_command(
    executable: str | Path,
    module: str,
    arguments: Iterable[str],
    root: str | Path | None = None,
) -> list[str]:
    """Normalize a Python module command for reproducible, private metadata."""

    command = [Path(executable).name or "python", "-m", module]
    for argument in arguments:
        token = str(argument)
        command.append(portable_path(token, root=root) if Path(token).is_absolute() else token)
    return command


def _normalized_relative_path(path: Path, root: Path) -> str:
    """Return a Unicode-normalized, POSIX path relative to ``root``."""

    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Path escapes the permitted root: {path}") from error
    return unicodedata.normalize("NFC", relative.as_posix())


def _require_regular_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required provenance file does not exist: {path}")
    if path.is_symlink():
        raise ValueError(f"Provenance inputs must not be symbolic links: {path}")
    if not path.is_file():
        raise ValueError(f"Required provenance path is not a regular file: {path}")


def _hash_file(path: Path) -> tuple[str, int]:
    """Hash a regular file and reject a concurrent mutation."""

    _require_regular_file(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise RuntimeError(f"Provenance input changed while it was hashed: {path}")
    return digest.hexdigest(), before.st_size


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a required regular file.

    Missing files, directories, and symbolic links fail closed instead of
    yielding a partial fingerprint.
    """

    return _hash_file(Path(path))[0]


def _update_framed(digest: Any, value: str | bytes) -> None:
    data = value.encode("utf-8") if isinstance(value, str) else value
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_include_path(root: Path, value: str | Path) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Source-tree include paths must be project-relative: {value}")
    candidate = root / relative
    # Resolve even when an optional candidate is absent so a symlinked parent
    # cannot escape the project root.
    _normalized_relative_path(candidate, root)
    return candidate


def _iter_directory_files(directory: Path) -> Iterable[Path]:
    for current, directory_names, file_names in os.walk(directory, followlinks=False):
        current_path = Path(current)
        kept_directories = []
        for name in sorted(directory_names):
            child = current_path / name
            normalized_name = name.casefold()
            if (
                normalized_name in _EXCLUDED_SOURCE_DIRECTORIES
                or normalized_name.endswith(".egg-info")
            ):
                continue
            if child.is_symlink():
                raise ValueError(f"Source-tree inputs must not contain symbolic links: {child}")
            kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in sorted(file_names):
            path = current_path / name
            if name in _EXCLUDED_SOURCE_FILENAMES:
                continue
            if path.suffix.casefold() in _EXCLUDED_SOURCE_SUFFIXES:
                continue
            if path.is_symlink():
                raise ValueError(f"Source-tree inputs must not contain symbolic links: {path}")
            if not path.is_file():
                raise ValueError(f"Source-tree input is not a regular file: {path}")
            yield path


def source_tree_fingerprint(
    project_root: str | Path,
    *,
    required_paths: Sequence[str | Path] = DEFAULT_SOURCE_TREE_PATHS,
    optional_paths: Sequence[str | Path] = OPTIONAL_SOURCE_TREE_PATHS,
) -> dict[str, Any]:
    """Fingerprint the executable source tree using an explicit allowlist.

    ``required_paths`` are fail-closed. Optional packaging files are included
    whenever present. Top-level ``data/``, ``results/``, caches, figures, and
    logs are excluded by construction rather than by a fragile broad glob.
    """

    root = Path(project_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project root does not exist: {root}")

    included: dict[str, Path] = {}

    def include(path: Path) -> None:
        relative_path = _normalized_relative_path(path, root)
        previous = included.get(relative_path)
        if previous is not None and previous.resolve() != path.resolve():
            raise ValueError(
                "Distinct source files have the same normalized portable path: "
                f"{previous} and {path}"
            )
        included[relative_path] = path

    for value in required_paths:
        candidate = _validate_include_path(root, value)
        if not candidate.exists():
            raise FileNotFoundError(f"Required source-tree path does not exist: {candidate}")
        if candidate.is_symlink():
            raise ValueError(f"Source-tree inputs must not be symbolic links: {candidate}")
        if candidate.is_file():
            paths: Iterable[Path] = (candidate,)
        elif candidate.is_dir():
            paths = _iter_directory_files(candidate)
        else:
            raise ValueError(f"Source-tree input is neither file nor directory: {candidate}")
        for path in paths:
            include(path)

    for value in optional_paths:
        candidate = _validate_include_path(root, value)
        if not candidate.exists():
            continue
        if candidate.is_symlink():
            raise ValueError(f"Source-tree inputs must not be symbolic links: {candidate}")
        if candidate.is_file():
            optional_files: Iterable[Path] = (candidate,)
        elif candidate.is_dir():
            optional_files = _iter_directory_files(candidate)
        else:
            raise ValueError(f"Source-tree input is neither file nor directory: {candidate}")
        for path in optional_files:
            include(path)

    if not included:
        raise ValueError("Source-tree allowlist contains no files")

    records: list[dict[str, Any]] = []
    tree_digest = hashlib.sha256()
    _update_framed(tree_digest, b"eptnet-source-tree-v1")
    for relative_path in sorted(included, key=lambda item: item.encode("utf-8")):
        file_sha256, size = _hash_file(included[relative_path])
        record = {"path": relative_path, "bytes": size, "sha256": file_sha256}
        records.append(record)
        _update_framed(tree_digest, relative_path)
        _update_framed(tree_digest, str(size))
        _update_framed(tree_digest, file_sha256)
    return {
        "sha256": tree_digest.hexdigest(),
        "file_count": len(records),
        "files": records,
    }


def source_tree_sha256(
    project_root: str | Path,
    *,
    required_paths: Sequence[str | Path] = DEFAULT_SOURCE_TREE_PATHS,
    optional_paths: Sequence[str | Path] = OPTIONAL_SOURCE_TREE_PATHS,
) -> str:
    """Return only the deterministic source-tree SHA-256."""

    return source_tree_fingerprint(
        project_root,
        required_paths=required_paths,
        optional_paths=optional_paths,
    )["sha256"]


def _read_manifest(path: Path) -> tuple[bytes, list[Mapping[str, Any]]]:
    _require_regular_file(path)
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise RuntimeError(f"Manifest changed while it was read: {path}")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"Manifest is not valid UTF-8: {path}") from error

    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"Manifest contains a blank record at line {line_number}: {path}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {error.msg}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Manifest record must be an object at {path}:{line_number}")
        records.append(record)
    if not records:
        raise ValueError(f"Manifest contains no records: {path}")
    return raw, records


def _portable_tensor_reference(reference: str) -> PurePosixPath:
    if not reference or "\x00" in reference:
        raise ValueError("Manifest tensor_file must be a non-empty portable path")
    if "\\" in reference:
        raise ValueError(f"Manifest tensor_file must use POSIX separators: {reference!r}")
    posix = PurePosixPath(reference)
    windows = PureWindowsPath(reference)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ValueError(f"Manifest tensor_file must be relative: {reference!r}")
    return posix


def manifest_fingerprint(
    manifest_path: str | Path,
    *,
    artifact_root: str | Path | None = None,
    tensor_field: str = "tensor_file",
) -> dict[str, Any]:
    """Hash a JSONL manifest and its tensors in manifest order.

    The combined identity includes raw manifest bytes plus each referenced
    tensor's portable path, size, and content digest. If ``artifact_root`` is
    supplied, every tensor must resolve inside it; traversal outside that root
    fails closed.
    """

    manifest = Path(manifest_path).resolve()
    root = Path(artifact_root).resolve() if artifact_root is not None else None
    if root is not None and not root.is_dir():
        raise FileNotFoundError(f"Artifact root does not exist: {root}")
    if root is not None:
        _normalized_relative_path(manifest, root)
    raw_manifest, records = _read_manifest(manifest)
    manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()

    tensor_records: list[dict[str, Any]] = []
    hash_cache: dict[Path, tuple[str, int]] = {}
    tensors_digest = hashlib.sha256()
    combined_digest = hashlib.sha256()
    _update_framed(tensors_digest, b"eptnet-ordered-tensors-v1")
    _update_framed(combined_digest, b"eptnet-manifest-and-tensors-v1")
    _update_framed(combined_digest, manifest_sha256)
    _update_framed(combined_digest, str(len(raw_manifest)))

    for index, record in enumerate(records):
        reference = record.get(tensor_field)
        if not isinstance(reference, str):
            raise ValueError(
                f"Manifest record {index + 1} has no string {tensor_field!r}: {manifest}"
            )
        posix_reference = _portable_tensor_reference(reference)
        tensor_path = manifest.parent.joinpath(*posix_reference.parts).resolve()
        if root is not None:
            portable_tensor_path = _normalized_relative_path(tensor_path, root)
        else:
            # This is lexical and therefore portable; it never serializes the
            # absolute parent of the manifest.
            portable_tensor_path = unicodedata.normalize("NFC", posix_reference.as_posix())
        if tensor_path not in hash_cache:
            hash_cache[tensor_path] = _hash_file(tensor_path)
        tensor_sha256, size = hash_cache[tensor_path]
        tensor_record = {
            "index": index,
            "path": portable_tensor_path,
            "bytes": size,
            "sha256": tensor_sha256,
        }
        tensor_records.append(tensor_record)
        for digest in (tensors_digest, combined_digest):
            _update_framed(digest, str(index))
            _update_framed(digest, portable_tensor_path)
            _update_framed(digest, str(size))
            _update_framed(digest, tensor_sha256)

    path_root = root if root is not None else manifest.parent
    return {
        "path": portable_path(manifest, root=path_root),
        "manifest_sha256": manifest_sha256,
        "manifest_bytes": len(raw_manifest),
        "tensor_count": len(tensor_records),
        "tensors_sha256": tensors_digest.hexdigest(),
        "manifest_and_tensors_sha256": combined_digest.hexdigest(),
        "tensor_files": tensor_records,
    }


def manifest_and_tensors_sha256(
    manifest_path: str | Path,
    *,
    artifact_root: str | Path | None = None,
    tensor_field: str = "tensor_file",
) -> str:
    """Return only the ordered manifest-and-tensor SHA-256."""

    return manifest_fingerprint(
        manifest_path,
        artifact_root=artifact_root,
        tensor_field=tensor_field,
    )["manifest_and_tensors_sha256"]


def required_artifact_fingerprints(
    dataset_root: str | Path,
    requirements_lock_path: str | Path,
    *,
    path_root: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Hash the schema, statistics, summary, and dependency lock file."""

    dataset = Path(dataset_root).resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset}")
    display_root = Path(path_root).resolve() if path_root is not None else dataset
    paths = {name: dataset / filename for name, filename in REQUIRED_ARTIFACT_FILENAMES.items()}
    paths["requirements_lock"] = Path(requirements_lock_path).resolve()

    fingerprints: dict[str, dict[str, Any]] = {}
    for name in sorted(paths):
        sha256, size = _hash_file(paths[name])
        fingerprints[name] = {
            "path": portable_path(paths[name], root=display_root),
            "bytes": size,
            "sha256": sha256,
        }
    return fingerprints


def git_state(project_root: str | Path) -> dict[str, Any] | None:
    """Return the current Git commit and scoped dirty flag when available.

    Git metadata is supplementary and is accepted only when ``project_root``
    itself is the repository root.  Git normally searches parent directories;
    without this guard, a standalone research artifact nested below an
    unrelated repository could silently inherit the parent's commit identity.
    Missing Git, a repository without a commit, a mismatched repository root,
    command failure, or timeout returns ``None``; callers still retain the
    authoritative source-tree content hash.
    """

    root = Path(project_root).resolve()

    def run(arguments: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            capture_output=True,
            timeout=5,
        )

    try:
        top_level_result = run(("rev-parse", "--show-toplevel"))
        if top_level_result.returncode != 0:
            return None
        encoded_repository_root = top_level_result.stdout.strip()
        if not encoded_repository_root:
            return None
        try:
            repository_root = Path(os.fsdecode(encoded_repository_root)).resolve()
        except (TypeError, ValueError, OSError):
            return None
        if os.path.normcase(str(repository_root)) != os.path.normcase(str(root)):
            return None

        commit_result = run(("rev-parse", "--verify", "HEAD^{commit}"))
        if commit_result.returncode != 0:
            return None
        try:
            commit = commit_result.stdout.strip().decode("ascii")
        except UnicodeDecodeError:
            return None
        if len(commit) < 40 or any(
            character not in "0123456789abcdefABCDEF" for character in commit
        ):
            return None
        status_result = run(("status", "--porcelain=v1", "--untracked-files=normal", "--", "."))
        if status_result.returncode != 0:
            return None
    except (
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
        subprocess.TimeoutExpired,
        OSError,
    ):
        return None
    return {"commit": commit.lower(), "dirty": bool(status_result.stdout.strip())}


def _discover_manifests(dataset_root: Path) -> list[Path]:
    manifest_directory = dataset_root / "manifests"
    if not manifest_directory.is_dir():
        raise FileNotFoundError(f"Manifest directory does not exist: {manifest_directory}")
    manifests = sorted(
        (path.resolve() for path in manifest_directory.glob("*.jsonl")),
        key=lambda path: _normalized_relative_path(path, dataset_root).encode("utf-8"),
    )
    if not manifests:
        raise FileNotFoundError(f"No JSONL manifests found in: {manifest_directory}")
    return manifests


def build_provenance_fingerprint(
    project_root: str | Path,
    dataset_root: str | Path,
    *,
    manifest_paths: Iterable[str | Path] | None = None,
    requirements_lock_path: str | Path | None = None,
    include_git: bool = True,
    source_paths: Sequence[str | Path] = DEFAULT_SOURCE_TREE_PATHS,
    optional_source_paths: Sequence[str | Path] = OPTIONAL_SOURCE_TREE_PATHS,
) -> dict[str, Any]:
    """Build a deterministic, portable, fail-closed provenance record.

    Relative manifest paths are interpreted below ``dataset_root``. Manifest
    records are sorted by portable manifest path, while tensor order inside
    each manifest is preserved. ``provenance_sha256`` depends only on content
    identities and portable logical paths, not on Git availability or absolute
    machine paths.
    """

    project = Path(project_root).resolve()
    dataset = Path(dataset_root).resolve()
    lock = (
        Path(requirements_lock_path).resolve()
        if requirements_lock_path is not None
        else project / "requirements-lock.txt"
    )
    source_tree = source_tree_fingerprint(
        project,
        required_paths=source_paths,
        optional_paths=optional_source_paths,
    )
    artifacts = required_artifact_fingerprints(dataset, lock, path_root=project)

    if manifest_paths is None:
        manifests = _discover_manifests(dataset)
    else:
        manifests = []
        for value in manifest_paths:
            candidate = Path(value)
            manifests.append(
                (candidate if candidate.is_absolute() else dataset / candidate).resolve()
            )
        if not manifests:
            raise ValueError("At least one manifest is required")
        manifests.sort(key=lambda path: _normalized_relative_path(path, dataset).encode("utf-8"))

    manifest_records = [manifest_fingerprint(path, artifact_root=dataset) for path in manifests]
    identity = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "source_tree_sha256": source_tree["sha256"],
        "required_artifacts": {
            name: record["sha256"] for name, record in sorted(artifacts.items())
        },
        "manifests": [
            {
                "path": record["path"],
                "manifest_and_tensors_sha256": record["manifest_and_tensors_sha256"],
            }
            for record in manifest_records
        ],
    }
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "provenance_sha256": _canonical_sha256(identity),
        "source_tree": source_tree,
        "git": git_state(project) if include_git else None,
        "required_artifacts": artifacts,
        "manifests": manifest_records,
    }
