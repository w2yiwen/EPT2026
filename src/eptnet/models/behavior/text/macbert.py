from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..common import EncodedSequence

MACBERT_MODEL_ID = "hfl/chinese-macbert-base"
MACBERT_MODEL_REVISION = "a986e004d2a7f2a1c2f5a3edef4e20604a974ed1"
MACBERT_SOURCE_COMMIT = "9a72882bf097fb2edd2ea5adeb4e0330b374ac2c"


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class MacBERTBaseEncoder:
    """Frozen HFL MacBERT-base with non-special-token masked-mean pooling."""

    output_dim = 768

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 128,
        allow_remote: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_length < 4:
            raise ValueError("max_length must be at least 4")
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        import torch
        from transformers import BertModel, BertTokenizerFast

        self._torch = torch
        self.device = _resolve_device(device)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        local_assets = Path(__file__).resolve().parent / "macbert_base" / "assets"
        requested = (
            Path(model_path).expanduser().resolve() if model_path is not None else local_assets
        )
        if (requested / "config.json").is_file():
            source = str(requested)
            load_options = {"local_files_only": True}
        elif allow_remote:
            source = MACBERT_MODEL_ID
            load_options = {"revision": MACBERT_MODEL_REVISION}
        else:
            raise FileNotFoundError(
                f"MacBERT assets are missing from {requested}; run the model asset installer"
            )
        self.tokenizer = BertTokenizerFast.from_pretrained(source, **load_options)
        self.model = BertModel.from_pretrained(source, **load_options).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def encode(self, texts: Sequence[str]) -> EncodedSequence:
        output = np.zeros((len(texts), self.output_dim), dtype=np.float32)
        mask = np.asarray([bool(text.strip()) for text in texts], dtype=bool)
        non_empty = np.flatnonzero(mask).tolist()
        for start in range(0, len(non_empty), self.batch_size):
            indices = non_empty[start : start + self.batch_size]
            batch_text = [texts[index] for index in indices]
            tokens = self.tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            special_tokens_mask = tokens.pop("special_tokens_mask")
            tokens = {name: value.to(self.device) for name, value in tokens.items()}
            special_tokens_mask = special_tokens_mask.to(self.device)
            with self._torch.inference_mode():
                hidden = self.model(**tokens).last_hidden_state
            weights = tokens["attention_mask"].bool() & ~special_tokens_mask.bool()
            denominator = weights.sum(dim=1, keepdim=True)
            pooled = (hidden * weights.unsqueeze(-1).to(hidden.dtype)).sum(dim=1)
            pooled = pooled / denominator.clamp_min(1).to(hidden.dtype)
            empty_content = denominator.squeeze(1) == 0
            if empty_content.any():
                pooled[empty_content] = hidden[empty_content, 0]
            output[np.asarray(indices)] = pooled.float().cpu().numpy()
        output[~mask] = 0.0
        return EncodedSequence(output, mask).validate(output_dim=self.output_dim)
