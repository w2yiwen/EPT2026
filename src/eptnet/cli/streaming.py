"""Simple prepare/normalize/infer entry points for the native-time protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from eptnet.config import load_config


def main():
    parser = argparse.ArgumentParser(description="EPT-Net native-stream inputs and inference")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare one synchronized recording")
    prepare.add_argument("--config", default="configs/paper.yaml")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--device")
    normalize = commands.add_parser("normalize", help="Fit training-partition physiology statistics")
    normalize.add_argument("--config", default="configs/paper.yaml")
    normalize.add_argument("--output")
    infer = commands.add_parser("infer", help="Load a checkpoint and process an independent recording")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--input", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--device", default="auto")
    infer.add_argument("--trace", action="store_true", help="Include per-entry Gaussian weights and support times")
    args = parser.parse_args()
    if args.command == "infer":
        from eptnet.evaluation.streaming import infer as run_inference
        result = run_inference(args.checkpoint, args.input, args.output, args.device, args.trace)
    else:
        config = load_config(args.config)
        if config["data"].get("input_mode") != "native_streams":
            raise ValueError("Use configs/paper.yaml for native-time preparation")
        if args.command == "prepare":
            from eptnet.data.prepare import prepare_file
            from eptnet.training.trainer import resolve_device
            device = str(resolve_device(args.device or config["training"]["device"]))
            result = prepare_file(args.input, args.output, config, device)
        else:
            from eptnet.data.streaming import normalization_statistics
            from eptnet.training.streaming import stream_datasets
            dataset = stream_datasets(config)["train"]
            path = Path(args.output or config["data"]["normalization_stats"])
            if path.exists():
                raise FileExistsError(f"Training normalization already exists: {path}")
            result = normalization_statistics(dataset)
            result["manifest_sha256"] = hashlib.sha256(dataset.path.read_bytes()).hexdigest()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, allow_nan=False)
            result = {"output": str(path), "training_subjects": result["subjects"]}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
