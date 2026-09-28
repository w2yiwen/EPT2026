from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

ALIGNMENT_SCHEMA_VERSION = 1


def load_video_audio(path: Path, sample_rate: int = 8_000) -> np.ndarray:
    """Decode a video's first audio stream without requiring an ffmpeg executable."""
    import av

    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return np.empty(0, dtype=np.float32)
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
        for frame in container.decode(stream):
            converted = resampler.resample(frame)
            for output in converted if isinstance(converted, list) else [converted]:
                if output is not None:
                    chunks.append(np.asarray(output.to_ndarray(), dtype=np.float32).reshape(-1))
        flushed = resampler.resample(None)
        for output in flushed if isinstance(flushed, list) else [flushed]:
            if output is not None:
                chunks.append(np.asarray(output.to_ndarray(), dtype=np.float32).reshape(-1))
    return np.nan_to_num(
        np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_characters(text: str) -> list[tuple[int, str]]:
    output: list[tuple[int, str]] = []
    for original_index, character in enumerate(text):
        normalized = unicodedata.normalize("NFKC", character).casefold()
        for item in normalized:
            category = unicodedata.category(item)
            if category[0] in {"L", "N"}:
                output.append((original_index, item))
    return output


def whisper_timed_characters(result: Mapping[str, object]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for segment in result.get("segments", []):
        words = segment.get("words") or []
        if not words and str(segment.get("text", "")).strip():
            words = [
                {
                    "word": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "probability": segment.get("avg_logprob"),
                }
            ]
        for word in words:
            characters = _normalized_characters(str(word.get("word", "")))
            if not characters:
                continue
            start = float(word["start"])
            end = max(start, float(word["end"]))
            duration = end - start
            for index, (_, character) in enumerate(characters):
                output.append(
                    {
                        "character": character,
                        "start": start + duration * index / len(characters),
                        "end": start + duration * (index + 1) / len(characters),
                        "score": word.get("probability"),
                    }
                )
    return output


def build_reference_timeline(
    turns: Sequence[Mapping[str, object]],
    annotated_turns: Sequence[Mapping[str, object]],
    whisper_result: Mapping[str, object],
    *,
    audio_duration_seconds: float,
    step_seconds: float = 1.0,
    minimum_reference_coverage: float = 0.70,
) -> dict[str, object]:
    if len(turns) != len(annotated_turns):
        raise ValueError("Annotated and text-only transcript passes disagree")
    if audio_duration_seconds <= 0 or step_seconds <= 0:
        raise ValueError("Audio duration and step size must be positive")

    speakers: Counter[str] = Counter()
    reference: list[dict[str, object]] = []
    for turn_index, (turn, annotated) in enumerate(zip(turns, annotated_turns, strict=True)):
        if (turn["speaker_id"], turn["text"]) != (annotated["speaker_id"], annotated["text"]):
            raise ValueError("Annotated and text-only transcript contents disagree")
        marks = annotated.get("marks")
        if not isinstance(marks, list) or len(marks) != len(str(turn["text"])):
            raise ValueError("Annotated transcript marks are missing or malformed")
        for character_index, normalized in _normalized_characters(str(turn["text"])):
            speaker = str(turn["speaker_id"])
            speakers[speaker] += 1
            reference.append(
                {
                    "turn_index": turn_index,
                    "character_index": character_index,
                    "character": normalized,
                    "original": str(turn["text"])[character_index],
                    "speaker_id": speaker,
                    "marked": bool(marks[character_index]),
                }
            )
    if not reference:
        raise ValueError("Transcript contains no alignable characters")
    ranked = sorted(speakers.items(), key=lambda item: (-item[1], int(item[0])))
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        raise ValueError("Target speaker is ambiguous")
    target_speaker = ranked[0][0]

    hypothesis = whisper_timed_characters(whisper_result)
    matcher = SequenceMatcher(
        None,
        [item["character"] for item in reference],
        [item["character"] for item in hypothesis],
        autojunk=False,
    )
    matched: dict[int, int] = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            matched[block.a + offset] = block.b + offset
    reference_coverage = len(matched) / len(reference)
    hypothesis_coverage = len(matched) / max(1, len(hypothesis))
    if reference_coverage < minimum_reference_coverage:
        raise ValueError(
            f"Whisper/reference exact-character coverage {reference_coverage:.3%} is below "
            f"the required {minimum_reference_coverage:.0%}"
        )

    num_steps = max(16, int(math.ceil(audio_duration_seconds / step_seconds)))
    step_text: list[list[str]] = [[] for _ in range(num_steps)]
    target_counts = np.zeros(num_steps, dtype=np.int64)
    other_counts = np.zeros(num_steps, dtype=np.int64)
    marked_counts = np.zeros(num_steps, dtype=np.int64)
    aligned_characters: list[dict[str, object]] = []
    for reference_index, hypothesis_index in matched.items():
        ref = reference[reference_index]
        timed = hypothesis[hypothesis_index]
        midpoint = (float(timed["start"]) + float(timed["end"])) / 2
        if midpoint < 0 or midpoint >= audio_duration_seconds:
            continue
        step = min(int(midpoint // step_seconds), num_steps - 1)
        step_text[step].append(str(ref["original"]))
        is_target = ref["speaker_id"] == target_speaker
        if is_target:
            target_counts[step] += 1
            if bool(ref["marked"]):
                marked_counts[step] += 1
        else:
            other_counts[step] += 1
        aligned_characters.append(
            {
                "reference_index": reference_index,
                "turn_index": ref["turn_index"],
                "character_index": ref["character_index"],
                "character": ref["original"],
                "start": timed["start"],
                "end": timed["end"],
                "score": timed["score"],
            }
        )

    occupied = (target_counts + other_counts) > 0
    target_mask = (target_counts > 0) & (target_counts > other_counts)
    labels = np.where(marked_counts > 0, 0, 1).astype(np.int64)
    return {
        "schema_version": ALIGNMENT_SCHEMA_VERSION,
        "status": "complete",
        "step_seconds": step_seconds,
        "audio_duration_seconds": audio_duration_seconds,
        "num_steps": num_steps,
        "step_text": ["".join(characters) for characters in step_text],
        "labels": labels.tolist(),
        "target_mask": target_mask.tolist(),
        "aligned_characters": aligned_characters,
        "statistics": {
            "alignment": "openai_whisper_word_dtw_plus_exact_reference_character_mapping",
            "target_speaker_id": target_speaker,
            "reference_characters": len(reference),
            "whisper_characters": len(hypothesis),
            "matched_reference_characters": len(matched),
            "reference_character_coverage": reference_coverage,
            "hypothesis_character_coverage": hypothesis_coverage,
            "target_valid_steps": int(target_mask.sum()),
            "target_valid_fraction": float(target_mask.mean()),
            "occupied_steps": int(occupied.sum()),
            "unmatched_reference_characters_fail_closed": len(reference) - len(matched),
        },
    }


def estimate_video_audio_offset(
    external_audio: np.ndarray,
    video_audio: np.ndarray,
    *,
    sample_rate: int,
    maximum_offset_seconds: float = 30.0,
) -> tuple[float, float]:
    """Return ``video_time = external_audio_time + offset`` and correlation."""
    from scipy.signal import correlate, correlation_lags

    block = max(1, sample_rate // 50)

    def envelope(signal: np.ndarray) -> np.ndarray:
        signal = np.asarray(signal, dtype=np.float32)
        usable = len(signal) // block * block
        if usable < block:
            raise ValueError("Audio is too short for offset estimation")
        values = np.sqrt(np.mean(signal[:usable].reshape(-1, block) ** 2, axis=1) + 1e-12)
        values = values - np.median(values)
        scale = np.std(values)
        return values / scale if scale > 1e-8 else values

    external = envelope(external_audio)
    video = envelope(video_audio)
    correlation = correlate(video, external, mode="full", method="fft")
    lags = correlation_lags(len(video), len(external), mode="full")
    limit = int(round(maximum_offset_seconds * 50))
    allowed = np.abs(lags) <= limit
    selected_index = np.flatnonzero(allowed)[int(np.argmax(correlation[allowed]))]
    lag = int(lags[selected_index])
    if lag >= 0:
        video_overlap = video[lag:]
        external_overlap = external[: len(video_overlap)]
    else:
        external_overlap = external[-lag:]
        video_overlap = video[: len(external_overlap)]
    length = min(len(video_overlap), len(external_overlap))
    if length < 10:
        raise ValueError("Audio overlap is too short for offset estimation")
    score = float(np.corrcoef(video_overlap[:length], external_overlap[:length])[0, 1])
    if not math.isfinite(score):
        raise ValueError("Audio offset correlation is not finite")
    return lag / 50.0, score


def load_alignment(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != ALIGNMENT_SCHEMA_VERSION or payload.get("status") != "complete":
        raise ValueError(f"Invalid Whisper alignment artifact: {path}")
    return payload
