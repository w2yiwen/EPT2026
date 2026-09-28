#!/usr/bin/env python3
"""Install pinned behavior encoder assets beneath src/eptnet/models/behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = CODE_ROOT.parent
MODEL_ROOT = CODE_ROOT / "src" / "eptnet" / "models" / "behavior"
CANDIDATE_HF_CACHE = REPOSITORY_ROOT / "third_party" / "research_candidates" / "hf_cache" / "hub"

WAVLM_REVISION = "4c66d4806a428f2e922ccfa1a962776e232d487b"
MACBERT_REVISION = "a986e004d2a7f2a1c2f5a3edef4e20604a974ed1"
WAVLM_SOURCE_COMMIT = "31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd"
MACBERT_SOURCE_COMMIT = "9a72882bf097fb2edd2ea5adeb4e0330b374ac2c"
OPENFACE_SOURCE_COMMIT = "3d4b5cf8d96138be42bed229447f36cbb09a5a29"
OPENFACE_RELEASE_URL = (
    "https://github.com/TadasBaltrusaitis/OpenFace/releases/download/"
    "OpenFace_2.2.0/OpenFace_2.2.0_win_x64.zip"
)
MARLIN_CHECKPOINT_URL = (
    "https://github.com/ControlNet/MARLIN/releases/download/model_v1/"
    "marlin_vit_small_ytf.encoder.pt"
)
MARLIN_CHECKPOINT_SIZE = 89_977_023
MARLIN_CHECKPOINT_SHA256 = "50fe50bfb1e1afbcccd2660f8267ac27469dabd8c9774159abc4772c1a1ac2d8"
OPENFACE_PATCH_ASSETS = {
    "cen_patches_0.25_of.dat": (
        60_602_360,
        "99d3df9888115428075de7b8f1bc86176881de640c2fdf47e2ab0cee556661bf",
    ),
    "cen_patches_0.35_of.dat": (
        60_602_360,
        "1070bff51b077ee6a18ce8e1ebbc6f568e1a7a469f6911a8263413878b3df469",
    ),
    "cen_patches_0.50_of.dat": (
        154_289_792,
        "3f08124067e326e83e3261a55c79493e745b01138c9c1b59713dd9b17424168e",
    ),
    "cen_patches_1.00_of.dat": (
        154_289_792,
        "bed961bbfa2cc41a709d44c1d1d8a92b2537bc46f2fd6ab430c137fedd0e21b0",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "EPT-Net-baseline-builder"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def _install_file(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(source) != _sha256(destination):
            raise FileExistsError(f"Refusing to overwrite mismatched asset: {destination}")
        return "existing"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _hf_snapshot(
    *,
    cache_name: str,
    model_id: str,
    revision: str,
    filenames: tuple[str, ...],
    destination: Path,
) -> list[dict[str, object]]:
    cached = CANDIDATE_HF_CACHE / cache_name / "snapshots" / revision
    if not cached.is_dir():
        from huggingface_hub import snapshot_download

        cached = Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                allow_patterns=list(filenames),
            )
        )
    records = []
    for filename in filenames:
        source = cached / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / filename
        method = _install_file(source, target)
        records.append(
            {
                "path": target.relative_to(CODE_ROOT).as_posix(),
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
                "installation": method,
            }
        )
    return records


def _install_vendor_sources() -> list[dict[str, object]]:
    downloads = {
        MODEL_ROOT / "audio" / "wavlm_base_plus" / "vendor" / "WavLM.py": (
            f"https://raw.githubusercontent.com/microsoft/unilm/{WAVLM_SOURCE_COMMIT}/wavlm/WavLM.py"
        ),
        MODEL_ROOT / "audio" / "wavlm_base_plus" / "vendor" / "modules.py": (
            f"https://raw.githubusercontent.com/microsoft/unilm/{WAVLM_SOURCE_COMMIT}/wavlm/modules.py"
        ),
        MODEL_ROOT / "audio" / "wavlm_base_plus" / "vendor" / "LICENSE": (
            f"https://raw.githubusercontent.com/microsoft/unilm/{WAVLM_SOURCE_COMMIT}/LICENSE"
        ),
        MODEL_ROOT / "text" / "macbert_base" / "vendor" / "README_EN.md": (
            f"https://raw.githubusercontent.com/ymcui/MacBERT/{MACBERT_SOURCE_COMMIT}/README_EN.md"
        ),
        MODEL_ROOT / "text" / "macbert_base" / "vendor" / "LICENSE": (
            f"https://raw.githubusercontent.com/ymcui/MacBERT/{MACBERT_SOURCE_COMMIT}/LICENSE"
        ),
        MODEL_ROOT / "face" / "openface" / "vendor" / "FeatureExtraction.cpp": (
            "https://raw.githubusercontent.com/TadasBaltrusaitis/OpenFace/"
            f"{OPENFACE_SOURCE_COMMIT}/exe/FeatureExtraction/FeatureExtraction.cpp"
        ),
        MODEL_ROOT / "face" / "openface" / "vendor" / "OpenFace-license.txt": (
            "https://raw.githubusercontent.com/TadasBaltrusaitis/OpenFace/"
            f"{OPENFACE_SOURCE_COMMIT}/OpenFace-license.txt"
        ),
    }
    records = []
    for destination, url in downloads.items():
        if not destination.exists():
            _download(url, destination)
        records.append(
            {
                "path": destination.relative_to(CODE_ROOT).as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
                "source": url,
            }
        )
    return records


def _install_openface_runtime() -> list[dict[str, object]]:
    assets = MODEL_ROOT / "face" / "openface" / "assets"
    if not list(assets.rglob("FeatureExtraction.exe")):
        with tempfile.TemporaryDirectory(prefix="eptnet-openface-") as temporary:
            archive = Path(temporary) / "OpenFace_2.2.0_win_x64.zip"
            _download(OPENFACE_RELEASE_URL, archive)
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(assets / "runtime")
    executable = next(iter(sorted(assets.rglob("FeatureExtraction.exe"))), None)
    if executable is None:
        raise FileNotFoundError("OpenFace release did not contain FeatureExtraction.exe")
    runtime_root = executable.parent
    patch_root = runtime_root / "model" / "patch_experts"
    invalid_patch_assets = [
        name
        for name, (expected_size, expected_sha256) in OPENFACE_PATCH_ASSETS.items()
        if not (patch_root / name).is_file()
        or (patch_root / name).stat().st_size != expected_size
        or _sha256(patch_root / name) != expected_sha256
    ]
    if invalid_patch_assets:
        for name in invalid_patch_assets:
            candidate = patch_root / name
            if candidate.is_file():
                candidate.unlink()
        downloader = runtime_root / "download_models.ps1"
        if os.name != "nt" or not downloader.is_file():
            raise RuntimeError(
                "The pinned OpenFace runtime requires its official Windows model downloader"
            )
        subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(downloader),
            ],
            cwd=runtime_root,
            check=True,
            timeout=900,
        )

    records = [
        {
            "path": executable.relative_to(CODE_ROOT).as_posix(),
            "bytes": executable.stat().st_size,
            "sha256": _sha256(executable),
            "redistributable": False,
        }
    ]
    for name, (expected_size, expected_sha256) in OPENFACE_PATCH_ASSETS.items():
        path = patch_root / name
        if not path.is_file() or path.stat().st_size != expected_size:
            raise RuntimeError(f"OpenFace model download is incomplete: {path}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"OpenFace model hash mismatch: {path}")
        records.append(
            {
                "path": path.relative_to(CODE_ROOT).as_posix(),
                "bytes": expected_size,
                "sha256": actual_sha256,
                "redistributable": False,
            }
        )
    return records


def _install_marlin_checkpoint() -> list[dict[str, object]]:
    destination = (
        MODEL_ROOT / "face" / "marlin" / "assets" / "marlin_vit_small_ytf.encoder.pt"
    )
    if not destination.is_file():
        _download(MARLIN_CHECKPOINT_URL, destination)
    if destination.stat().st_size != MARLIN_CHECKPOINT_SIZE:
        raise RuntimeError(f"MARLIN checkpoint size mismatch: {destination}")
    actual_sha256 = _sha256(destination)
    if actual_sha256 != MARLIN_CHECKPOINT_SHA256:
        raise RuntimeError(f"MARLIN checkpoint hash mismatch: {destination}")
    return [
        {
            "path": destination.relative_to(CODE_ROOT).as_posix(),
            "bytes": MARLIN_CHECKPOINT_SIZE,
            "sha256": actual_sha256,
            "source": MARLIN_CHECKPOINT_URL,
            "redistributable": False,
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-openface-runtime", action="store_true")
    args = parser.parse_args()

    records: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "vendor_sources": _install_vendor_sources(),
        "wavlm": _hf_snapshot(
            cache_name="models--microsoft--wavlm-base-plus",
            model_id="microsoft/wavlm-base-plus",
            revision=WAVLM_REVISION,
            filenames=("config.json", "preprocessor_config.json", "pytorch_model.bin"),
            destination=MODEL_ROOT / "audio" / "wavlm_base_plus" / "assets",
        ),
        "macbert": _hf_snapshot(
            cache_name="models--hfl--chinese-macbert-base",
            model_id="hfl/chinese-macbert-base",
            revision=MACBERT_REVISION,
            filenames=(
                "added_tokens.json",
                "config.json",
                "pytorch_model.bin",
                "special_tokens_map.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "vocab.txt",
            ),
            destination=MODEL_ROOT / "text" / "macbert_base" / "assets",
        ),
    }
    records["openface"] = [] if args.skip_openface_runtime else _install_openface_runtime()
    records["marlin"] = _install_marlin_checkpoint()
    output = MODEL_ROOT / "local_asset_installation.json"
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False))


if __name__ == "__main__":
    main()
