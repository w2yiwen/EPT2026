"""Data entry point for inventorying the read-only 927 video drop.

The source directory is never modified.  Results are written to a JSON file in
the desktop project so that subsequent pairing decisions are reproducible.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import av
from faster_whisper import WhisperModel

VIDEO_NAME_RE = re.compile(r"^VID_?(?P<date>\d{8})_?(?P<time>\d{6})\.mp4$", re.IGNORECASE)


def parse_recorded_at(path: Path) -> str | None:
    match = VIDEO_NAME_RE.match(path.name)
    if not match:
        return None
    value = datetime.strptime(match.group("date") + match.group("time"), "%Y%m%d%H%M%S")
    return value.isoformat(timespec="seconds")


def inspect_media(path: Path) -> dict[str, object]:
    with av.open(str(path), mode="r") as container:
        duration_seconds = (
            float(container.duration / av.time_base) if container.duration is not None else None
        )
        streams: list[dict[str, object]] = []
        for stream in container.streams:
            payload: dict[str, object] = {
                "index": stream.index,
                "type": stream.type,
                "codec": stream.codec_context.name,
                "duration_seconds": (
                    float(stream.duration * stream.time_base)
                    if stream.duration is not None and stream.time_base is not None
                    else None
                ),
            }
            if stream.type == "video":
                payload.update(
                    {
                        "width": stream.codec_context.width,
                        "height": stream.codec_context.height,
                        "average_rate": str(stream.average_rate) if stream.average_rate else None,
                    }
                )
            elif stream.type == "audio":
                payload.update(
                    {
                        "sample_rate": stream.codec_context.sample_rate,
                        "channels": stream.codec_context.channels,
                    }
                )
            streams.append(payload)
        return {
            "duration_seconds": duration_seconds,
            "container_metadata": dict(container.metadata),
            "streams": streams,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--model", default="base")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--clip-start", type=float, default=0.0)
    parser.add_argument("--clip-end", type=float, default=240.0)
    parser.add_argument("--include-pattern")
    parser.add_argument("--disable-hotwords", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve(strict=True)
    videos = sorted(source_root.glob("*.mp4"), key=lambda path: (parse_recorded_at(path) or "", path.name))
    if args.include_pattern:
        include_re = re.compile(args.include_pattern, re.IGNORECASE)
        videos = [path for path in videos if include_re.search(path.name)]
    if not videos:
        raise SystemExit(f"No MP4 files found in {source_root}")

    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        download_root=str(args.download_root),
        local_files_only=args.local_files_only,
    )
    results: list[dict[str, object]] = []
    hotwords = (
        "中文访谈 自我介绍 姓名 年龄 专业 学校 家庭 朋友 经历 "
        "AJY DYN FZC HSY LFY LMX LZY WHZ WQ WYB XFX YPJ YWJ ZHJ ZSR ZYL ZYX "
        "张禾佳 赵银星 李茂西 王一白 谢丰霞"
    )

    for index, path in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] {path.name}", flush=True)
        media = inspect_media(path)
        segments_iter, info = model.transcribe(
            str(path),
            language="zh",
            beam_size=5,
            vad_filter=True,
            clip_timestamps=f"{args.clip_start:g},{args.clip_end:g}",
            condition_on_previous_text=False,
            initial_prompt="中文访谈，自我介绍，包括姓名、年龄、专业、学校、家庭、朋友、经历。",
            hotwords=None if args.disable_hotwords else hotwords,
        )
        segments = [
            {
                "start_seconds": round(segment.start, 3),
                "end_seconds": round(segment.end, 3),
                "text": segment.text.strip(),
                "average_log_probability": round(segment.avg_logprob, 6),
                "no_speech_probability": round(segment.no_speech_prob, 6),
            }
            for segment in segments_iter
        ]
        results.append(
            {
                "source_path": str(path),
                "source_name": path.name,
                "bytes": path.stat().st_size,
                "recorded_at_from_filename": parse_recorded_at(path),
                **media,
                "transcription": {
                    "model": args.model,
                    "language": info.language,
                    "language_probability": round(info.language_probability, 6),
                    "clip_start_seconds": args.clip_start,
                    "clip_end_seconds": args.clip_end,
                    "segments": segments,
                    "text": " ".join(segment["text"] for segment in segments),
                },
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"source_root": str(source_root), "videos": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
