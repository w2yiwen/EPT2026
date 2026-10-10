from .baselines import GateHUB, LSTR, MulT, TeSTra
from .eptnet import EPTNet, EPTNetConfig


def build_model(config):
    if config.get("data", {}).get("input_mode") == "native_streams":
        from .streaming import EPTNet as StreamingEPTNet

        return StreamingEPTNet(config)
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
    if name == "mult":
        return MulT(model_config)
    raise ValueError(f"Unknown model name: {name}")


def load_model(checkpoint, device="cpu"):
    """Load either tensor protocol through the same checkpoint interface."""
    import torch

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(payload["config"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.checkpoint_metadata = {"epoch": payload.get("epoch"),
                                 "provenance": payload.get("provenance")}
    model.eval()
    return model


__all__ = [
    "EPTNet",
    "EPTNetConfig",
    "GateHUB",
    "LSTR",
    "MulT",
    "TeSTra",
    "build_model",
    "load_model",
]
