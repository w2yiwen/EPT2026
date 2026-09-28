from .baselines import EarlyFusionGRU, FusionTransformer
from .behavior import (
    CausalTCNDecoder,
    MacBERTBaseEncoder,
    OpenFaceFeatureEncoder,
    WavLMBasePlusEncoder,
)
from .eptnet import EPTNet, EPTNetConfig
from .physiology import (
    EEGNetFeatureEncoder,
    EEGNetPreprocessor,
    EEGNetV4,
    NeuroKitPPGEncoder,
    PPGEncodedSequence,
)


def build_model(config):
    model_config = EPTNetConfig.from_mapping(config)
    name = config.get("model", config).get("name", "eptnet")
    if name == "eptnet":
        return EPTNet(model_config)
    if name == "early_fusion_gru":
        return EarlyFusionGRU(model_config)
    if name == "fusion_transformer":
        return FusionTransformer(model_config)
    raise ValueError(f"Unknown model name: {name}")


__all__ = [
    "EPTNet",
    "EPTNetConfig",
    "EarlyFusionGRU",
    "FusionTransformer",
    "build_model",
    "CausalTCNDecoder",
    "MacBERTBaseEncoder",
    "OpenFaceFeatureEncoder",
    "WavLMBasePlusEncoder",
    "EEGNetFeatureEncoder",
    "EEGNetPreprocessor",
    "EEGNetV4",
    "NeuroKitPPGEncoder",
    "PPGEncodedSequence",
]
