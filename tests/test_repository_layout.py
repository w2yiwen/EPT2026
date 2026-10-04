from __future__ import annotations

from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
EXECUTABLE_SUFFIXES = {".py", ".ps1", ".sh"}
ARTIFACT_DIRECTORIES = ("results", "figures", "logs")
SCRIPT_ROOT = CODE_ROOT / "scripts"
SCRIPT_RESPONSIBILITIES = {"audit", "experiments", "reporting"}


def _executable_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path.relative_to(CODE_ROOT)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in EXECUTABLE_SUFFIXES
    )


@pytest.mark.parametrize("directory", ARTIFACT_DIRECTORIES)
def test_generated_artifact_directories_contain_no_executable_source(directory: str) -> None:
    assert _executable_files(CODE_ROOT / directory) == []


def test_project_root_contains_no_ad_hoc_launchers() -> None:
    launchers = sorted(
        path.name
        for path in PROJECT_ROOT.iterdir()
        if path.is_file() and path.suffix.casefold() in EXECUTABLE_SUFFIXES
    )
    assert launchers == []


def test_data_root_is_not_traversed_and_has_no_top_level_source() -> None:
    data_root = CODE_ROOT / "data"
    # Public/source-only clones intentionally omit the ignored private data root.
    if not data_root.exists():
        return
    top_level_source = sorted(
        path.name
        for path in data_root.iterdir()
        if path.is_file() and path.suffix.casefold() in EXECUTABLE_SUFFIXES
    )
    assert top_level_source == []


def test_script_root_is_an_index_only() -> None:
    assert sorted(path.name for path in SCRIPT_ROOT.iterdir() if path.is_file()) == [
        "README.md"
    ]
    assert {
        path.name for path in SCRIPT_ROOT.iterdir() if path.is_dir()
    } == SCRIPT_RESPONSIBILITIES


def test_report_and_figure_generators_live_under_scripts() -> None:
    expected = {
        SCRIPT_ROOT / "reporting" / "gen_fig_training.py",
    }
    assert all(path.is_file() for path in expected)
    assert not (CODE_ROOT / "results" / "gen_fig_training.py").exists()


def test_baselines_use_one_model_per_module() -> None:
    baseline_root = CODE_ROOT / "src" / "eptnet" / "models" / "baselines"
    assert (baseline_root / "lstr.py").is_file()
    assert (baseline_root / "gatehub.py").is_file()
    assert (baseline_root / "testra.py").is_file()
    assert not (baseline_root / "early_fusion_gru.py").exists()
    assert not (baseline_root / "fusion_transformer.py").exists()
    assert (baseline_root / "common.py").is_file()
    assert not (baseline_root.parent / "baselines.py").exists()
