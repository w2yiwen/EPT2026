from pathlib import Path

import pytest
import yaml

from eptnet.config import load_config


def _write_config(tmp_path, config, name="config.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return str(path)


def _set_nested(config, path, value):
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def test_ablation_inherits_default():
    config = load_config("configs/ablation_fixed_reader.yaml")
    assert config["model"]["adaptive_reader"] is False
    assert config["model"]["use_text"] is False
    assert config["model"]["hidden_dim"] > 0
    assert config["data"]["input_mode"] == "precomputed_features"
    assert config["data"]["train_manifest"].endswith("manifests/train.jsonl")
    assert config["training"]["sequence_protocol"] == "continuous_session"
    assert config["training"]["batch_size"] == 1


def test_with_text_audit_changes_only_the_intended_protocol_fields():
    trusted = load_config("configs/default.yaml")
    audit = load_config("configs/ablation_with_text.yaml")
    assert trusted["model"]["use_text"] is False
    assert audit["model"]["use_text"] is True
    assert audit["experiment"]["name"] == "eptnet_with_text_continuous_audit"
    assert audit["experiment"]["output_dir"] == "results/eptnet_with_text_continuous_audit"

    trusted["model"]["use_text"] = True
    trusted["experiment"]["name"] = audit["experiment"]["name"]
    trusted["experiment"]["output_dir"] = audit["experiment"]["output_dir"]
    assert trusted == audit


def test_all_trusted_configs_use_isolated_no_text_outputs():
    explicit_text_configs = {
        "ablation_with_text.yaml",
        "bci_subjects_official_text.yaml",
        "eptnet_v6_behavior12_seed42.yaml",
            "eptnet_v6_marlin12_4060_seed42.yaml",
            "eptnet_v6_marlin12_seed42.yaml",
            "eptnet_v6_marlin11_4060_seed42.yaml",
            "eptnet_v6_marlin11_4060_windowed_seed42.yaml",
            "baseline_early_fusion_gru_marlin11_4060_windowed_seed42.yaml",
    }
    for path in Path("configs").glob("*.yaml"):
        if path.name in explicit_text_configs:
            continue
        raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw_config, dict) or not (
            "experiment" in raw_config or "inherits" in raw_config
        ):
            # Registries for frozen feature extractors share this directory but are
            # not executable EPT-Net training configurations.
            continue
        config = load_config(str(path))
        assert config["model"]["use_text"] is False, path
        assert config["experiment"]["name"].endswith("_no_text"), path
        assert config["experiment"]["output_dir"].endswith("_no_text"), path


def test_official_text_config_uses_separate_macbert_dataset_and_output():
    config = load_config("configs/bci_subjects_official_text.yaml")
    assert config["model"]["use_text"] is True
    assert config["model"]["use_audio"] is False
    assert config["data"]["dataset_name"] == "bci_subjects_ept_v2_macbert"
    assert "macbert" in config["experiment"]["name"]
    assert "macbert" in config["experiment"]["output_dir"]


def test_seed42_v6_config_is_the_gated_current_dataset_entry_point():
    config = load_config("configs/eptnet_v6_seed42.yaml")
    assert config["experiment"]["seed"] == 42
    assert config["data"]["dataset_name"] == (
        "bci_subjects_ept_v6_session_registered_no_facial"
    )
    assert config["model"]["state_update"] == "gated"
    assert config["model"]["use_video"] is False
    assert config["model"]["use_audio"] is False
    assert config["model"]["use_text"] is False


def test_seed42_behavior12_config_enables_all_behavior_modalities():
    config = load_config("configs/eptnet_v6_behavior12_seed42.yaml")
    assert config["data"]["dataset_name"] == "bci_subjects_ept_v6_behavior_complete12"
    assert (config["data"]["video_dim"], config["data"]["audio_dim"], config["data"]["text_dim"]) == (
        215,
        768,
        768,
    )
    assert config["model"]["use_behavior_context"] is True
    assert config["model"]["use_video"] is True
    assert config["model"]["use_audio"] is True
    assert config["model"]["use_text"] is True
    assert config["model"]["use_eeg_time"] is True
    assert config["model"]["use_eeg_spec"] is True
    assert config["model"]["use_hr"] is True


def test_seed42_marlin12_config_enables_all_modalities_with_marlin_dimension():
    config = load_config("configs/eptnet_v6_marlin12_seed42.yaml")
    assert config["data"]["dataset_name"] == "bci_subjects_ept_v6_marlin_complete12"
    assert (config["data"]["video_dim"], config["data"]["audio_dim"], config["data"]["text_dim"]) == (
        384,
        768,
        768,
    )
    assert all(
        config["model"][key]
        for key in (
            "use_behavior_context",
            "use_video",
            "use_audio",
            "use_text",
            "use_eeg_time",
            "use_eeg_spec",
            "use_hr",
        )
    )


def test_seed42_marlin12_4060_config_is_fully_isolated() -> None:
    config = load_config("configs/eptnet_v6_marlin12_4060_seed42.yaml")
    assert config["data"]["dataset_name"] == "bci_subjects_ept_v6_marlin4060_complete12"
    assert "marlin4060" in config["data"]["train_manifest"]
    assert config["experiment"]["output_dir"] == (
        "results/eptnet_v6_marlin12_4060_gated_seed42"
    )
    assert (config["data"]["video_dim"], config["data"]["audio_dim"], config["data"]["text_dim"]) == (
        384,
        768,
        768,
    )
    assert all(
        config["model"][key]
        for key in ("use_video", "use_audio", "use_text", "use_eeg_time", "use_eeg_spec", "use_hr")
    )


def test_seed42_marlin11_4060_config_declares_exact_aligned_cohort() -> None:
    config = load_config("configs/eptnet_v6_marlin11_4060_seed42.yaml")
    expected = config["data"]["expected_session_ids"]
    assert len(expected) == len(set(expected)) == 11
    assert "session_011" not in expected
    assert config["data"]["excluded_session_ids"] == ["session_011"]
    assert config["data"]["require_aligned_behavior_modalities"] is True
    assert config["data"]["dataset_name"] == "bci_subjects_ept_v6_marlin4060_aligned11"
    assert config["experiment"]["output_dir"] == (
        "results/eptnet_v6_marlin11_4060_gated_seed42"
    )


def test_marlin11_windowed_and_gru_configs_use_causal_training_only() -> None:
    main = load_config("configs/eptnet_v6_marlin11_4060_windowed_seed42.yaml")
    baseline = load_config(
        "configs/baseline_early_fusion_gru_marlin11_4060_windowed_seed42.yaml"
    )
    for config in (main, baseline):
        assert config["training"]["sequence_protocol"] == "causal_windows"
        assert config["training"]["window_size"] == 128
        assert config["training"]["window_stride"] == 32
        assert config["training"]["window_warmup_steps"] == 32
        assert config["data"]["dataset_name"] == "bci_subjects_ept_v6_marlin4060_aligned11"
    assert main["model"]["name"] == "eptnet"
    assert baseline["model"]["name"] == "early_fusion_gru"


@pytest.mark.parametrize(
    ("location", "unknown_key"),
    (
        ((), "trianing"),
        (("model",), "hidden_dims"),
        (("model", "cache_lengths"), "times"),
        (("training", "scheduler"), "factors"),
    ),
)
def test_load_config_rejects_unknown_keys(tmp_path, location, unknown_key):
    config = load_config("configs/default.yaml")
    target = config
    for key in location:
        target = target[key]
    target[unknown_key] = 1

    with pytest.raises(ValueError, match=unknown_key):
        load_config(_write_config(tmp_path, config))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("experiment", "seed"), -1, "seed"),
        (("model", "hidden_dim"), 0, "hidden_dim"),
        (("model", "hidden_dim"), 63, "divisible"),
        (("model", "num_heads"), 0, "num_heads"),
        (("model", "dropout"), 1.0, "dropout"),
        (("model", "cache_lengths", "time"), 0, "cache_lengths.time"),
        (("model", "min_read_width"), 0.0, "min_read_width"),
        (("model", "fixed_read_width"), 0.01, "fixed_read_width"),
        (("model", "state_update"), "unknown", "state_update"),
        (("loss", "label_smoothing"), 1.0, "label_smoothing"),
        (("training", "epochs"), 0, "epochs"),
        (("training", "batch_size"), 2, "batch_size"),
        (("training", "learning_rate"), 0.0, "learning_rate"),
        (("training", "weight_decay"), -0.1, "weight_decay"),
        (("evaluation", "frame_threshold"), 1.1, "frame_threshold"),
        (
            ("evaluation", "calibration_min_threshold"),
            0.95,
            "calibration_min_threshold",
        ),
    ),
)
def test_load_config_rejects_invalid_core_values(tmp_path, path, value, message):
    config = load_config("configs/default.yaml")
    _set_nested(config, path, value)

    with pytest.raises(ValueError, match=message):
        load_config(_write_config(tmp_path, config))


def test_load_config_rejects_non_mapping_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(str(path))
