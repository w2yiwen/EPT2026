from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from eptnet.models.behavior import CausalTCNDecoder, OpenFaceFeatureEncoder
from eptnet.models.behavior.face.openface import OPENFACE_FRAME_COLUMNS


def test_openface_encoder_pools_interpretable_frame_features(tmp_path: Path) -> None:
    path = tmp_path / "openface.csv"
    columns = ["timestamp", "confidence", "success", *OPENFACE_FRAME_COLUMNS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for frame, timestamp in enumerate((0.1, 0.6, 1.1, 1.6)):
            row = {
                "timestamp": timestamp,
                "confidence": 0.95,
                "success": 1,
            }
            row.update(
                {name: frame + index / 100 for index, name in enumerate(OPENFACE_FRAME_COLUMNS)}
            )
            writer.writerow(row)

    encoder = OpenFaceFeatureEncoder(executable=None)
    encoded = encoder.encode_csv(path, num_steps=3)

    assert encoded.features.shape == (3, 215)
    assert encoded.mask.tolist() == [True, True, False]
    assert np.all(encoded.features[2] == 0)
    assert np.isfinite(encoded.features).all()


def test_openface_encoder_keeps_frame_on_exact_second_boundary(tmp_path: Path) -> None:
    path = tmp_path / "boundary.csv"
    columns = ["timestamp", "confidence", "success", *OPENFACE_FRAME_COLUMNS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for timestamp in (0.0, 1.0):
            row = {"timestamp": timestamp, "confidence": 0.95, "success": 1}
            row.update({name: 0.5 for name in OPENFACE_FRAME_COLUMNS})
            writer.writerow(row)

    encoded = OpenFaceFeatureEncoder().encode_csv(path)

    assert encoded.features.shape == (2, 215)
    assert encoded.mask.tolist() == [True, True]


def test_openface_encoder_fails_closed_on_wrong_csv_schema(tmp_path: Path) -> None:
    path = tmp_path / "not_openface.csv"
    path.write_text("timestamp,confidence,success\n0.1,0.99,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        OpenFaceFeatureEncoder().encode_csv(path)


def test_causal_tcn_decoder_shapes_and_prefix_invariance() -> None:
    torch.manual_seed(7)
    decoder = CausalTCNDecoder(12, hidden_dim=16, dropout=0.0).eval()
    features = torch.randn(2, 7, 12)
    mask = torch.ones(2, 7, dtype=torch.bool)
    changed = features.clone()
    changed[:, 5:] += 100.0

    with torch.inference_mode():
        output = decoder(features, mask)
        changed_output = decoder(changed, mask)

    assert output["class_logits"].shape == (2, 7, 2)
    assert output["boundary_logits"].shape == (2, 7, 2)
    assert output["offsets"].shape == (2, 7, 2)
    torch.testing.assert_close(
        output["class_logits"][:, :5],
        changed_output["class_logits"][:, :5],
        rtol=0,
        atol=1e-6,
    )
