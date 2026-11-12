# OpenRestor audio restoration template

This starter project implements the planned OpenRestor container interface. The included model is an **identity baseline**: it copies each degraded input to the output and does not restore audio. Replace `model.restore`, install your model dependencies, and make your trained weights available to the container before submitting a model.

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
mkdir -p outputs
docker run --rm --network none \
  -v /path/to/degraded:/input/degraded:ro \
  -v "$PWD/outputs:/output" \
  openrestor-template:local
```

## Connecting a trained model

`model.restore(audio, sample_rate, metadata)` receives one degraded stereo waveform at a time and must return a finite `float32` array with the same shape. `run_inference.py` handles manifest IDs, output WAVs, timing, and the output manifest. Your model implementation may load weights baked into the image or from a read-only mounted directory such as `/weights`; declare any additional environment variables in your own project. If your model requires a GPU, build its CUDA runtime into your image and request GPU access when running it.

The ARIEL SA3 pilot exercised this contract with a trained LoRA inside its own GPU Docker image. It produced eight outputs accepted and scored by OpenRestor on a **private validation set**. That test used mounted SA3, SAME-L, and LoRA weights. This starter image still contains only the identity baseline; the pilot weights and audio are not distributed here.

Before sharing a trained container, run it on an allowed validation package and pass the `validate-restored` command above. Pin the image by digest, record the inference command and model asset versions, and keep test-only data out of training. The public dataset release and official submission terms are still being finalized. See the project [guide](https://github.com/vita-epfl/OpenRestor/blob/main/docs/openrestor_guide.md) for the current contract.
