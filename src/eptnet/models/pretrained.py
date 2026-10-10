"""Frozen behavioral encoders and prefix-only feature extraction."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FrozenBehaviorEncoders(nn.Module):
    """Lazy optional dependencies; pretrained weights load only on construction."""

    def __init__(self, config: Mapping, device: str = "cpu") -> None:
        super().__init__()
        from marlin_pytorch import Marlin
        from huggingface_hub import snapshot_download
        from speechbrain.inference.interfaces import Pretrained
        from speechbrain.utils.fetching import LocalStrategy
        from transformers import AutoModel, AutoTokenizer

        self.settings = dict(config)
        video = config.get("video_checkpoint")
        self.video = (Marlin.from_file("marlin_vit_small_ytf", str(video)) if video
                      else Marlin.from_online("marlin_vit_small_ytf"))
        self.video_norm = nn.LayerNorm(384, elementwise_affine=False)
        audio_source = config.get("audio_source", "speechbrain/emotion-recognition-wav2vec2-IEMOCAP")
        audio_snapshot = (str(Path(audio_source)) if Path(audio_source).is_dir() else snapshot_download(
            repo_id=audio_source, revision=config.get("audio_revision"),
            cache_dir=str(Path(config.get("cache_dir", ".cache/eptnet")) / "hub"),
            allow_patterns=["hyperparams.yaml", "wav2vec2.ckpt", "model.ckpt", "label_encoder.txt"]))
        self.audio = Pretrained.from_hparams(
            source=audio_snapshot,
            overrides={"pretrained_path": audio_snapshot},
            savedir=str(Path(config.get("cache_dir", ".cache/eptnet")) / "audio" / Path(audio_snapshot).name),
            run_opts={"device": device},
            local_strategy=LocalStrategy.COPY)
        text_source = config.get("text_source", "hfl/chinese-macbert-base")
        self.tokenizer = AutoTokenizer.from_pretrained(text_source, revision=config.get("text_revision"))
        self.tokenizer.truncation_side = "left"
        self.text = AutoModel.from_pretrained(text_source, revision=config.get("text_revision"))
        def digest(path):
            value = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    value.update(block)
            return value.hexdigest()
        self.provenance = {"video_model": "marlin_vit_small_ytf",
            "video_sha256": digest(video or ".marlin/marlin_vit_small_ytf.encoder.pt"),
            "audio_source": audio_source, "audio_revision": config.get("audio_revision"),
            "audio_sha256": digest(Path(audio_snapshot) / "wav2vec2.ckpt"),
            "text_source": text_source, "text_revision": getattr(self.text.config, "_commit_hash", None)}
        self.to(device)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        # Parent model train() must never activate dropout in these backbones.
        return super().train(False)

    @property
    def device(self):
        return next(self.text.parameters()).device

    @torch.no_grad()
    def encode_video(self, clips: Tensor) -> Tensor:
        clips = clips.to(device=self.device, dtype=torch.float32)
        if clips.shape[1:3] != (3, 16) or clips.shape[-2:] != (224, 224):
            raise ValueError("MARLIN expects observed RGB facial clips [B,3,16,224,224] in [0,1]")
        tokens = self.video.extract_features(clips, keep_seq=True)
        return self.video_norm(tokens.mean(dim=1))

    @torch.no_grad()
    def encode_audio(self, waveforms: Tensor) -> Tensor:
        # The IEMOCAP-finetuned wav2vec2 module, before its emotion classifier.
        features = self.audio.mods.wav2vec2(waveforms.to(self.device, dtype=torch.float32))
        return features.mean(dim=1)

    @torch.no_grad()
    def encode_text(self, texts: Sequence[str]) -> Tensor:
        encoded = self.tokenizer(list(texts), padding=True, truncation=True, max_length=128,
                                 return_special_tokens_mask=True, return_tensors="pt")
        special = encoded.pop("special_tokens_mask").to(self.device).bool()
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        content = encoded["attention_mask"].bool() & ~special
        states = self.text(**encoded).last_hidden_state
        return (states * content[..., None]).sum(dim=1) / content.sum(dim=1, keepdim=True).clamp_min(1)

    @torch.no_grad()
    def prefix(self, now: float, *, frames: Tensor, frame_times: Tensor,
               audio: Tensor, audio_times: Tensor, words: Sequence[Mapping]) -> dict:
        """One decision: trailing face frames, preceding 2 s audio, completed words.

        Inputs are synchronized, facially cropped RGB frames [N,3,H,W], 16-kHz
        mono audio, and reference words with text/start/end. No full-file video
        extractor or full-transcript text descriptor enters this method.
        """
        output = {"video": torch.zeros(384, device=self.device),
                  "audio": torch.zeros(768, device=self.device),
                  "text": torch.zeros(768, device=self.device)}
        mask = torch.zeros(3, dtype=torch.bool, device=self.device)
        video_seconds = float(self.settings.get("video_window_seconds", 2.0))
        observed = torch.nonzero((frame_times <= now) & (frame_times > now - video_seconds)).flatten()
        if observed.numel() >= 16:
            positions = torch.linspace(0, observed.numel() - 1, 16).round().long()
            selected = frames[observed[positions]].float()
            selected = F.interpolate(selected, size=(224, 224), mode="bilinear", align_corners=False)
            if torch.isfinite(selected).all():
                output["video"] = self.encode_video(selected.permute(1, 0, 2, 3)[None])[0]
                mask[0] = True
        observed_audio = (audio_times > now - 2.0) & (audio_times <= now)
        waveform = audio[observed_audio]
        timestamps = audio_times[observed_audio]
        if (waveform.numel() == 32000 and torch.isfinite(waveform).all()
                and torch.all(torch.abs(timestamps[1:] - timestamps[:-1] - 1 / 16000) < 1e-5)):
            output["audio"] = self.encode_audio(waveform[None])[0]
            mask[1] = True
        completed = [str(word["text"]) for word in words if float(word["end"]) <= now]
        if completed:
            output["text"] = self.encode_text([" ".join(completed)])[0]
            mask[2] = True
        output["modality_mask"] = mask
        return output
