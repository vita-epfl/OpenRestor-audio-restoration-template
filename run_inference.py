"""OpenRestor container entry point for public and hidden degraded inputs."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from model import restore

SAMPLE_RATE = 44_100
CHANNELS = 2


def safe_input(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe degraded_path: {relative}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Input escapes root: {relative}")
    return resolved


def main() -> None:
    manifest = Path(os.environ.get("OPENRESTOR_INPUT_MANIFEST", "/input/degraded/index.jsonl"))
    input_root = Path(os.environ.get("OPENRESTOR_INPUT_ROOT", "/input/degraded/audio"))
    output_root = Path(os.environ.get("OPENRESTOR_OUTPUT_ROOT", "/output/audio"))
    output_manifest = Path(os.environ.get("OPENRESTOR_OUTPUT_MANIFEST", "/output/restoration_outputs.jsonl"))
    output_root.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    count = 0
    with manifest.open(encoding="utf-8") as source, output_manifest.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = row.get("id")
            if not isinstance(item_id, str) or not item_id or item_id in seen:
                raise ValueError(f"Missing or duplicate item ID on line {line_number}")
            seen.add(item_id)
            duration = row.get("duration_seconds")
            if row.get("sample_rate") != SAMPLE_RATE or row.get("channels") != CHANNELS or not isinstance(duration, (int, float)) or duration <= 0:
                raise ValueError(f"Noncanonical input metadata for {item_id}")
            path = safe_input(input_root, row["degraded_path"])
            audio, rate = sf.read(path, dtype="float32", always_2d=True)
            expected_samples = round(float(duration) * SAMPLE_RATE)
            if rate != SAMPLE_RATE or audio.shape != (expected_samples, CHANNELS) or not np.isfinite(audio).all():
                raise ValueError(f"Noncanonical input WAV for {item_id}")
            start = time.monotonic()
            restored = np.asarray(restore(audio, rate, row), dtype=np.float32)
            elapsed = time.monotonic() - start
            if restored.shape != audio.shape or not np.isfinite(restored).all():
                raise ValueError(f"Invalid model output for {item_id}")
            output_path = output_root / f"item-{count:06d}.wav"
            sf.write(output_path, restored, SAMPLE_RATE, format="WAV", subtype="PCM_24")
            result = {
                "id": item_id,
                "restored_audio_path": os.path.relpath(output_path, output_manifest.parent).replace(os.sep, "/"),
                "duration_sec": float(duration),
                "batch_time_seconds": elapsed,
            }
            target.write(json.dumps(result, sort_keys=True) + "\n")
            count += 1
    if count == 0:
        raise ValueError("Input manifest contains no items")
    print(f"Wrote {count} restored item(s) to {output_manifest}")


if __name__ == "__main__":
    main()
