from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from eptnet.config import load_config
from eptnet.data import validate_batch
from eptnet.decoding import decode_events
from eptnet.losses import EPTNetLoss
from eptnet.models import EPTNetConfig, build_model
from eptnet.models.reader import EventGuidedReader, PersistentMultimodalUpdate


def _config():
    config = load_config("configs/default.yaml")
    config["model"]["hidden_dim"] = 32
    config["model"]["num_heads"] = 4
    config["model"]["cache_lengths"] = {"time": 4, "spec": 4, "hr": 4}
    config["model"]["spectral_n_ffts"] = [16, 32]
    config["model"]["num_glimpses"] = 2
    config["data"]["eeg_channels"] = 2
    config["data"]["eeg_samples_per_step"] = 32
    config["data"]["hr_samples_per_step"] = 8
    config["data"]["eeg_time_dim"] = 4
    config["data"]["eeg_spectral_dim"] = 7
    config["data"]["physiology_dim"] = 6
    config["data"]["video_dim"] = 8
    config["data"]["audio_dim"] = 6
    config["data"]["text_dim"] = 5
    return config


def _batch(config):
    data = config["data"]
    batch_size, steps = 2, 4
    batch = {
        "video": torch.randn(batch_size, steps, data["video_dim"]),
        "audio": torch.randn(batch_size, steps, data["audio_dim"]),
        "text": torch.randn(batch_size, steps, data["text_dim"]),
        "sequence_mask": torch.ones(batch_size, steps, dtype=torch.bool),
        "modality_mask": torch.ones(batch_size, steps, 3, dtype=torch.bool),
        "labels": torch.ones(batch_size, steps, dtype=torch.long),
        "boundaries": torch.zeros(batch_size, steps, 2),
        "offsets": torch.zeros(batch_size, steps, 2),
        "positive_mask": torch.zeros(batch_size, steps, dtype=torch.bool),
    }
    if data["input_mode"] == "raw_windows":
        batch.update(
            {
                "eeg": torch.randn(
                    batch_size,
                    steps,
                    data["eeg_channels"],
                    data["eeg_samples_per_step"],
                ),
                "hr": torch.randn(
                    batch_size,
                    steps,
                    data["hr_channels"],
                    data["hr_samples_per_step"],
                ),
            }
        )
    else:
        batch.update(
            {
                "eeg_time": torch.randn(batch_size, steps, data["eeg_time_dim"]),
                "eeg_spectral": torch.randn(batch_size, steps, data["eeg_spectral_dim"]),
                "physiology": torch.randn(batch_size, steps, data["physiology_dim"]),
            }
        )
    # Production semantics are 0=deception (event-positive), 1=truth.
    batch["labels"][:, 1:4] = 0
    batch["positive_mask"][:, 1:4] = True
    batch["boundaries"][:, 1, 0] = 1.0
    batch["boundaries"][:, 3, 1] = 1.0
    batch["offsets"][:, 1:4, 0] = torch.tensor([0.0, 1.0, 2.0])
    batch["offsets"][:, 1:4, 1] = torch.tensor([2.0, 1.0, 0.0])
    validate_batch(batch, require_targets=True, input_mode=data["input_mode"])
    return batch


def test_forward_loss_and_backward():
    torch.manual_seed(0)
    config = _config()
    model = build_model(config)
    batch = _batch(config)
    outputs = model(batch)
    assert outputs["class_logits"].shape == (2, 4, 2)
    assert outputs["boundary_logits"].shape == (2, 4, 2)
    assert outputs["offsets"].shape == (2, 4, 2)
    assert outputs["read_centers"].shape == (2, 4, 3)
    assert outputs["write_gates"].shape == (2, 4, 32)
    assert outputs["retention_gates"].shape == (2, 4, 32)
    assert torch.all((outputs["write_gates"] >= 0) & (outputs["write_gates"] <= 1))
    assert torch.all((outputs["retention_gates"] >= 0) & (outputs["retention_gates"] <= 1))
    assert torch.all(outputs["offsets"] >= 0)
    losses = EPTNetLoss(config)(outputs, batch)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_loss_ignores_non_target_intervals():
    config = _config()
    batch = _batch(config)
    batch["target_mask"] = torch.tensor([[True, False, True, True], [True, False, True, True]])
    outputs = {
        "class_logits": torch.randn(2, 4, 2),
        "boundary_logits": torch.randn(2, 4, 2),
        "offsets": torch.rand(2, 4, 2),
    }
    changed = {key: value.clone() for key, value in outputs.items()}
    changed["class_logits"][:, 1] = torch.tensor([100.0, -100.0])
    changed["boundary_logits"][:, 1] = torch.tensor([100.0, -100.0])
    changed["offsets"][:, 1] = 100.0

    criterion = EPTNetLoss(config)
    original_loss = criterion(outputs, batch)["total"]
    changed_loss = criterion(changed, batch)["total"]

    torch.testing.assert_close(original_loss, changed_loss, rtol=0, atol=0)


def test_causal_prefix_invariance():
    torch.manual_seed(1)
    config = _config()
    model = build_model(config).eval()
    batch = _batch(config)
    changed = deepcopy(batch)
    changed["eeg_time"][:, 3] += 100.0
    changed["eeg_spectral"][:, 3] -= 100.0
    changed["physiology"][:, 3] += 25.0
    changed["video"][:, 3] += 50.0
    with torch.no_grad():
        original_output = model(batch)["class_logits"][:, :3]
        changed_output = model(changed)["class_logits"][:, :3]
    torch.testing.assert_close(original_output, changed_output, rtol=0, atol=1e-6)


@pytest.mark.parametrize("model_name", ("eptnet", "early_fusion_gru", "fusion_transformer"))
def test_all_models_are_causal_under_future_input_perturbations(model_name):
    torch.manual_seed(11)
    config = _config()
    config["model"]["name"] = model_name
    model = build_model(config).eval()
    batch = _batch(config)
    changed = deepcopy(batch)
    for name in ("eeg_time", "eeg_spectral", "physiology", "video", "audio", "text"):
        changed[name][:, 3] += 100.0
    with torch.no_grad():
        original_output = model(batch)["class_logits"][:, :3]
        changed_output = model(changed)["class_logits"][:, :3]
    torch.testing.assert_close(original_output, changed_output, rtol=0, atol=1e-6)


def test_fusion_transformer_position_encoding_is_dynamic_and_dtype_safe():
    config = _config()
    config["model"].update(name="fusion_transformer", dropout=0.0)
    model = build_model(config).eval()

    for dtype in (torch.float16, torch.float32, torch.float64):
        inputs = torch.zeros(2, 7, config["model"]["hidden_dim"], dtype=dtype)
        encoded = model.position_encoding(inputs)
        prefix = model.position_encoding(inputs[:, :4])
        assert encoded.shape == inputs.shape
        assert encoded.device == inputs.device
        assert encoded.dtype == dtype
        torch.testing.assert_close(encoded[:, :4], prefix, rtol=0, atol=0)
        assert not torch.equal(encoded[:, 0], encoded[:, 1])


def test_fusion_transformer_truncated_prefix_matches_full_sequence():
    torch.manual_seed(12)
    config = _config()
    config["model"].update(name="fusion_transformer", dropout=0.0)
    model = build_model(config).double().eval()
    batch = _batch(config)
    batch = {
        name: value.double()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
        else value
        for name, value in batch.items()
    }
    prefix_steps = 3
    prefix_batch = {
        name: value[:, :prefix_steps]
        if isinstance(value, torch.Tensor)
        and value.ndim >= 2
        and value.shape[1] == batch["sequence_mask"].shape[1]
        else value
        for name, value in batch.items()
    }

    with torch.no_grad():
        full_output = model(batch)
        prefix_output = model(prefix_batch)
    for name in ("class_logits", "boundary_logits", "offsets", "states"):
        torch.testing.assert_close(
            full_output[name][:, :prefix_steps],
            prefix_output[name],
            rtol=0,
            atol=1e-12,
        )


def test_batch_contract_rejects_non_prefix_sequence_mask():
    config = _config()
    batch = _batch(config)
    batch["sequence_mask"][0] = torch.tensor([True, False, True, False])
    with pytest.raises(ValueError, match="contiguous valid prefix"):
        validate_batch(batch, require_targets=True, input_mode=config["data"]["input_mode"])


def test_batch_contract_rejects_empty_sequence():
    config = _config()
    batch = _batch(config)
    batch["sequence_mask"][0] = False
    with pytest.raises(ValueError, match="at least one valid step"):
        validate_batch(batch, require_targets=True, input_mode=config["data"]["input_mode"])


def _assert_predictions_equal(original, changed):
    for name in ("class_logits", "boundary_logits", "offsets", "states"):
        torch.testing.assert_close(original[name], changed[name], rtol=0, atol=1e-6)


@pytest.mark.parametrize("model_name", ("eptnet", "early_fusion_gru", "fusion_transformer"))
def test_default_protocol_ignores_text_values(model_name):
    torch.manual_seed(2)
    config = _config()
    assert config["model"]["use_text"] is False
    config["model"]["name"] = model_name
    model = build_model(config).eval()
    batch = _batch(config)
    changed = deepcopy(batch)
    changed["text"] = torch.randn_like(changed["text"]) * 100.0
    with torch.no_grad():
        original_output = model(batch)
        changed_output = model(changed)
    _assert_predictions_equal(original_output, changed_output)


@pytest.mark.parametrize("model_name", ("eptnet", "early_fusion_gru", "fusion_transformer"))
def test_text_audit_switch_makes_text_observable(model_name):
    torch.manual_seed(3)
    config = _config()
    config["model"]["use_text"] = True
    config["model"]["name"] = model_name
    model = build_model(config).eval()
    batch = _batch(config)
    changed = deepcopy(batch)
    changed["text"] = changed["text"] + 10.0
    with torch.no_grad():
        original_logits = model(batch)["class_logits"]
        changed_logits = model(changed)["class_logits"]
    assert not torch.allclose(original_logits, changed_logits, rtol=0, atol=1e-6)


@pytest.mark.parametrize("model_name", ("eptnet", "early_fusion_gru", "fusion_transformer"))
def test_unavailable_text_token_cannot_affect_predictions(model_name):
    torch.manual_seed(4)
    config = _config()
    config["model"]["use_text"] = True
    config["model"]["name"] = model_name
    model = build_model(config).eval()
    batch = _batch(config)
    batch["modality_mask"][..., 2] = False
    changed = deepcopy(batch)
    changed["text"] = torch.randn_like(changed["text"]) * 100.0
    with torch.no_grad():
        original_output = model(batch)
        changed_output = model(changed)
    _assert_predictions_equal(original_output, changed_output)


@pytest.mark.parametrize("model_name", ("eptnet", "early_fusion_gru", "fusion_transformer"))
def test_unavailable_physiology_tokens_cannot_affect_predictions(model_name):
    torch.manual_seed(41)
    config = _config()
    config["model"]["name"] = model_name
    model = build_model(config).eval()
    batch = _batch(config)
    batch["physiology_mask"] = torch.ones(2, 4, 3, dtype=torch.bool)
    batch["physiology_mask"][:, 2, :] = False
    changed = deepcopy(batch)
    changed["eeg_time"][:, 2] += 1000.0
    changed["eeg_spectral"][:, 2] -= 1000.0
    changed["physiology"][:, 2] += 1000.0
    with torch.no_grad():
        original_output = model(batch)
        changed_output = model(changed)
    _assert_predictions_equal(original_output, changed_output)


def test_eptnet_all_missing_inputs_are_strictly_inert():
    torch.manual_seed(42)
    config = _config()
    model = build_model(config).eval()
    batch = _batch(config)
    batch["physiology_mask"] = torch.zeros(2, 4, 3, dtype=torch.bool)
    # The raw text stream satisfies the input contract, but text is disabled by
    # the trusted protocol.  Consequently no enabled behavior token exists.
    batch["modality_mask"] = torch.zeros(2, 4, 3, dtype=torch.bool)
    batch["modality_mask"][..., 2] = True
    changed = deepcopy(batch)
    for name in ("eeg_time", "eeg_spectral", "physiology", "video", "audio", "text"):
        changed[name] = torch.full_like(changed[name], torch.nan)
        changed[name].reshape(-1)[0] = torch.inf

    with torch.no_grad():
        original_output = model(batch)
        changed_output = model(changed)

    _assert_predictions_equal(original_output, changed_output)
    assert torch.equal(original_output["states"], torch.zeros_like(original_output["states"]))
    assert torch.isfinite(original_output["class_logits"]).all()
    for branch_weights in original_output["read_weights"].values():
        assert all(torch.equal(weights, torch.zeros_like(weights)) for weights in branch_weights)


def test_eptnet_accepts_a_real_timestep_with_every_modality_missing():
    torch.manual_seed(142)
    config = _config()
    model = build_model(config).eval()
    batch = _batch(config)
    batch["physiology_mask"] = torch.ones(2, 4, 3, dtype=torch.bool)
    batch["physiology_mask"][:, 2, :] = False
    batch["modality_mask"][:, 2, :] = False
    changed = deepcopy(batch)
    for name in ("eeg_time", "eeg_spectral", "physiology", "video", "audio", "text"):
        changed[name][:, 2] = torch.full_like(changed[name][:, 2], 10_000.0)

    with torch.no_grad():
        output = model(batch)
        changed_output = model(changed)

    assert torch.isfinite(output["class_logits"]).all()
    _assert_predictions_equal(output, changed_output)


def test_all_missing_physiology_matches_globally_disabled_branches():
    torch.manual_seed(43)
    config = _config()
    missing_model = build_model(config).eval()
    disabled_config = deepcopy(config)
    disabled_config["model"].update(use_eeg_time=False, use_eeg_spec=False, use_hr=False)
    disabled_model = build_model(disabled_config).eval()
    disabled_model.load_state_dict(missing_model.state_dict())

    missing_batch = _batch(config)
    missing_batch["physiology_mask"] = torch.zeros(2, 4, 3, dtype=torch.bool)
    disabled_batch = deepcopy(missing_batch)
    disabled_batch["physiology_mask"] = torch.ones(2, 4, 3, dtype=torch.bool)

    with torch.no_grad():
        missing_output = missing_model(missing_batch)
        disabled_output = disabled_model(disabled_batch)

    _assert_predictions_equal(missing_output, disabled_output)
    torch.testing.assert_close(
        missing_output["read_centers"], disabled_output["read_centers"], rtol=0, atol=0
    )
    for name in ("time", "spec", "hr"):
        for missing_weights, disabled_weights in zip(
            missing_output["read_weights"][name],
            disabled_output["read_weights"][name],
            strict=True,
        ):
            torch.testing.assert_close(missing_weights, disabled_weights, rtol=0, atol=0)


def test_no_enabled_behavior_matches_globally_disabled_context():
    torch.manual_seed(44)
    config = _config()
    missing_model = build_model(config).eval()
    disabled_config = deepcopy(config)
    disabled_config["model"]["use_behavior_context"] = False
    disabled_model = build_model(disabled_config).eval()
    disabled_model.load_state_dict(missing_model.state_dict())

    batch = _batch(config)
    batch["modality_mask"] = torch.zeros(2, 4, 3, dtype=torch.bool)
    batch["modality_mask"][..., 2] = True
    with torch.no_grad():
        missing_output = missing_model(batch)
        disabled_output = disabled_model(batch)

    _assert_predictions_equal(missing_output, disabled_output)
    torch.testing.assert_close(
        missing_output["read_centers"], disabled_output["read_centers"], rtol=0, atol=0
    )


def test_temporary_physiology_gap_reads_only_past_available_cache():
    torch.manual_seed(45)
    config = _config()
    config["model"]["cache_lengths"] = {"time": 2, "spec": 2, "hr": 2}
    model = build_model(config).eval()
    batch = _batch(config)
    batch["physiology_mask"] = torch.zeros(2, 4, 3, dtype=torch.bool)
    batch["physiology_mask"][:, 0] = True
    batch["modality_mask"] = torch.zeros(2, 4, 3, dtype=torch.bool)
    batch["modality_mask"][..., 2] = True

    with torch.no_grad():
        output = model(batch)

    for name in ("time", "spec", "hr"):
        gap_weights = output["read_weights"][name][1]
        torch.testing.assert_close(
            gap_weights[..., 0], torch.ones_like(gap_weights[..., 0]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            gap_weights[..., 1], torch.zeros_like(gap_weights[..., 1]), rtol=0, atol=0
        )
        assert torch.equal(
            output["read_weights"][name][2],
            torch.zeros_like(output["read_weights"][name][2]),
        )
    assert not torch.allclose(output["states"][:, 0], output["states"][:, 1])
    torch.testing.assert_close(output["states"][:, 1], output["states"][:, 2], rtol=0, atol=0)
    torch.testing.assert_close(output["states"][:, 2], output["states"][:, 3], rtol=0, atol=0)


def test_reader_all_false_cache_has_zero_evidence_and_read_weights():
    torch.manual_seed(46)
    reader = EventGuidedReader(
        hidden_dim=8,
        num_heads=2,
        num_glimpses=3,
        min_width=0.08,
        max_width=1.0,
        fixed_width=0.25,
        adaptive=True,
        shared_policy=False,
        latest_state_bypass=True,
        dropout=0.0,
    ).eval()
    previous_state = torch.randn(2, 8)
    behavior_context = torch.randn(2, 8)
    latest = {name: torch.randn(2, 8) for name in reader.BRANCHES}
    caches = {name: torch.randn(2, 4, 8) for name in reader.BRANCHES}
    cache_masks = {name: torch.zeros(2, 4, dtype=torch.bool) for name in reader.BRANCHES}

    with torch.no_grad():
        evidence, metadata = reader(previous_state, behavior_context, latest, caches, cache_masks)

    for name in reader.BRANCHES:
        assert torch.equal(evidence[name], torch.zeros_like(evidence[name]))
        assert torch.equal(metadata["weights"][name], torch.zeros_like(metadata["weights"][name]))
        assert torch.isfinite(evidence[name]).all()


@pytest.mark.parametrize("persistent", (True, False))
def test_update_preserves_state_for_row_with_no_available_tokens(persistent):
    torch.manual_seed(47)
    update = PersistentMultimodalUpdate(
        hidden_dim=8, num_heads=2, dropout=0.0, persistent=persistent
    ).eval()
    previous_state = torch.randn(2, 8)
    tokens = torch.randn(2, 4, 8)
    tokens[0] = torch.nan
    token_mask = torch.tensor([[False, False, False, False], [True, False, True, False]])

    with torch.no_grad():
        state = update(previous_state, tokens, token_mask)

    torch.testing.assert_close(state[0], previous_state[0], rtol=0, atol=0)
    assert torch.isfinite(state).all()


def test_gated_update_exposes_bounded_write_and_retention_controls():
    torch.manual_seed(48)
    update = PersistentMultimodalUpdate(
        hidden_dim=8,
        num_heads=2,
        dropout=0.0,
        persistent=True,
        state_update="gated",
    ).eval()
    previous_state = torch.randn(2, 8)
    tokens = torch.randn(2, 4, 8)
    token_mask = torch.tensor([[True, True, False, False], [False, False, False, False]])

    with torch.no_grad():
        state, metadata = update(
            previous_state,
            tokens,
            token_mask,
            return_metadata=True,
        )

    assert state.shape == previous_state.shape
    for name in ("write_gate", "retention_gate"):
        values = metadata[name]
        assert values.shape == previous_state.shape
        assert torch.all((values >= 0) & (values <= 1))
        assert torch.equal(values[1], torch.zeros_like(values[1]))
    torch.testing.assert_close(state[1], previous_state[1], rtol=0, atol=0)


@pytest.mark.parametrize("model_name", ("eptnet", "early_fusion_gru", "fusion_transformer"))
def test_all_behavior_modalities_disabled_are_finite_and_invariant(model_name):
    torch.manual_seed(5)
    config = _config()
    config["model"].update(use_video=False, use_audio=False, use_text=False)
    config["model"]["name"] = model_name
    model = build_model(config).eval()
    batch = _batch(config)
    changed = deepcopy(batch)
    for name in ("video", "audio", "text"):
        changed[name] = torch.randn_like(changed[name]) * 100.0
    with torch.no_grad():
        original_output = model(batch)
        changed_output = model(changed)
    assert torch.isfinite(original_output["states"]).all()
    _assert_predictions_equal(original_output, changed_output)


def test_model_config_spectral_fallback_matches_production_schema():
    fallback = EPTNetConfig.from_mapping({"data": {}, "model": {}})
    assert fallback.eeg_spectral_dim == 40
    assert fallback.use_text is False


def test_decoder_contract_uses_production_positive_class():
    outputs = {
        "class_logits": torch.tensor([[[-4.0, 4.0], [4.0, -4.0], [4.0, -4.0], [-4.0, 4.0]]]),
        "boundary_logits": torch.tensor([[[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [0.0, 0.0]]]),
        "offsets": torch.tensor([[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]]),
        "sequence_mask": torch.ones(1, 4, dtype=torch.bool),
    }
    decoded = decode_events(outputs)
    assert len(decoded) == 1 and len(decoded[0]) == 1
    assert decoded[0][0]["start"] <= decoded[0][0]["end"]
    assert decoded[0][0]["emit_step"] == 1.0


def test_baseline_output_contracts():
    config = _config()
    batch = _batch(config)
    for name in ("early_fusion_gru", "fusion_transformer"):
        config["model"]["name"] = name
        outputs = build_model(config)(batch)
        assert outputs["class_logits"].shape == (2, 4, 2)
        assert outputs["boundary_logits"].shape == (2, 4, 2)
        assert outputs["offsets"].shape == (2, 4, 2)


def test_padded_steps_remain_finite():
    config = _config()
    batch = _batch(config)
    batch["sequence_mask"][1, 2:] = False
    batch["modality_mask"][1, 2:] = False
    outputs = build_model(config)(batch)
    assert torch.isfinite(outputs["class_logits"]).all()
    torch.testing.assert_close(outputs["states"][1, 1], outputs["states"][1, 2])
    torch.testing.assert_close(outputs["states"][1, 2], outputs["states"][1, 3])


def test_raw_window_mode_is_preserved():
    config = _config()
    config["data"]["input_mode"] = "raw_windows"
    batch = _batch(config)
    outputs = build_model(config)(batch)
    assert outputs["class_logits"].shape == (2, 4, 2)
    assert torch.isfinite(outputs["class_logits"]).all()
