from .baselines import GateHUB, LSTR, TeSTra
from .eptnet import EPTNet, EPTNetConfig


def build_model(config):
    model_config = EPTNetConfig.from_mapping(config)
    name = config.get("model", config).get("name", "eptnet")
    if name == "eptnet":
        return EPTNet(model_config)
    if name == "lstr":
        return LSTR(model_config)
    if name == "gatehub":
        return GateHUB(model_config)
    if name == "testra":
        return TeSTra(model_config)
    raise ValueError(f"Unknown model name: {name}")


__all__ = [
    "EPTNet",
    "EPTNetConfig",
    "GateHUB",
    "LSTR",
    "TeSTra",
    "build_model",
]
