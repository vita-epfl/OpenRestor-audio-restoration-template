# OpenRestor audio restoration template

This starter project implements the planned OpenRestor container interface. The included model is an **identity baseline**: it copies each degraded input to the output and does not restore audio. Replace `model.restore` and bundle your weights before submitting a trained model.

## Input and output contract

Each input manifest row supplies `id`, `degraded_path`, `sample_rate`, `channels`, and `duration_seconds`. The entry point reads one 44.1 kHz stereo WAV per row and preserves its duration (30 seconds for the paired synthetic track; a real-audio test track may use another duration). It writes one WAV and one `restoration_outputs.jsonl` row with the same ID. It reads the four organizer environment variables `OPENRESTOR_INPUT_MANIFEST`, `OPENRESTOR_INPUT_ROOT`, `OPENRESTOR_OUTPUT_ROOT`, and `OPENRESTOR_OUTPUT_MANIFEST`. Relative `degraded_path` values resolve under the input root. The output manifest uses paths relative to itself.

The organizer does not mount clean references in the submission container. The published degradation metadata may guide training, but inference must handle the fields present in the supplied manifest. Do not expect hidden labels or reference audio.

## Local interface test

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
OPENRESTOR_INPUT_MANIFEST=/path/to/degraded/index.jsonl \
OPENRESTOR_INPUT_ROOT=/path/to/degraded/audio \
OPENRESTOR_OUTPUT_ROOT=outputs/audio \
OPENRESTOR_OUTPUT_MANIFEST=outputs/restoration_outputs.jsonl \
.venv/bin/python run_inference.py
```

Validate with the OpenRestor scorer when it is available:

```bash
openrestor-score validate-restored \
  --degraded-manifest /path/to/degraded/index.jsonl \
  --restored-manifest outputs/restoration_outputs.jsonl
```

## Container

```bash
docker build -t openrestor-template:local .
docker run --rm --network none \
  -v /path/to/degraded:/input/degraded:ro \
  -v "$PWD/outputs:/output" \
  openrestor-template:local
```

For the future official track, submit an immutable image digest, build recipe, model weights, and an inference command. The OpenRestor release and submission terms are still being finalized. See the project [guide](https://github.com/vita-epfl/OpenRestor/blob/main/docs/openrestor_guide.md) for the current contract.
