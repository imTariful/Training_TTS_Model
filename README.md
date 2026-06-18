# Bangladeshi Bengali F5-TTS Pipeline

Production-oriented pipeline for fine-tuning the official F5-TTS training stack on a custom multi-speaker Bangladeshi Bengali dataset.

## Features

- Dataset analysis with duplicate, missing-file, corruption, and metadata validation checks
- Audio preprocessing to 24 kHz mono WAV with trimming, normalization, and optional filtering
- Bangla text normalization with number, date, currency, and abbreviation handling
- Bangla phonemization with mixed English and Arabic loan-word support
- Quality filtering with configurable thresholds
- Official F5-TTS project preparation (`raw.arrow`, `duration.json`, `vocab.txt`, `metadata.csv`)
- ECAPA-TDNN speaker embeddings
- Single-GPU, multi-GPU, DeepSpeed, and Accelerate launch support
- Objective evaluation report generation
- Inference with speaker, emotion, dialect, and voice-cloning controls
- Export to checkpoint, safetensors, and Hugging Face format

## Project Structure

- `project/configs/` - configuration objects
- `project/preprocessing/` - dataset analysis and preprocessing scripts
- `project/training/` - training orchestration and experiment runner
- `project/evaluation/` - objective evaluation and HTML reports
- `project/inference/` - synthesis entry point
- `project/utils/` - shared helpers
- `project/tests/` - unit tests

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For training, install the upstream F5-TTS repository in editable mode or make its `f5_tts` package available on `PYTHONPATH`.

## Dataset Analysis

```bash
python -m project.preprocessing.analyze_dataset --metadata data/metadata.csv --report dataset_analysis.json
```

## Audio Preprocessing

```bash
python -m project.preprocessing.preprocess_audio --metadata data/metadata.csv --output-metadata processed_metadata.csv
```

## Text Normalization

```bash
python -m project.preprocessing.normalize_bn "১০/০৫/২০২৬ ৳৫০০"
```

## Phonemization

```bash
python -m project.preprocessing.phonemizer_bn "আজকের আবহাওয়া খুব সুন্দর"
```

## Prepare F5-TTS Project

```bash
python -m project.preprocessing.prepare_f5tts_dataset --metadata-file processed_metadata.csv --project-name my_speak --output-root data
```

This generates an F5-TTS compatible project folder at `data/my_speak/` with:

- `metadata.csv`
- `metadata_detailed.csv`
- `raw.arrow`
- `duration.json`
- `vocab.txt`
- `train.txt`
- `valid.txt`

## Speaker Embeddings

```bash
python -m project.preprocessing.extract_speaker_embeddings --metadata processed_metadata.csv --output speaker_embeddings.pt
```

## Training

```bash
python train_f5tts.py --project-name my_speak --exp-name F5TTS_v1_Base --dry-run
```

Remove `--dry-run` to launch `accelerate` against the official `f5_tts.train.finetune_cli` entrypoint. The launcher auto-detects GPU and recommends batch size and gradient accumulation based on VRAM.

### Recommended GPU Settings

- RTX 3060: batch size 4, gradient accumulation 4
- RTX 4060: batch size 2, gradient accumulation 8
- RTX 4070: batch size 4, gradient accumulation 4
- RTX 4090: batch size 8, gradient accumulation 2
- A100: batch size 16, gradient accumulation 1

## Conditioning Experiments

Run four experiments through `project.training.experiments`:

- A: text only
- B: text + speaker
- C: text + speaker + emotion
- D: text + speaker + emotion + dialect

## Evaluation

```bash
python evaluate.py --reference-dir references --synthesized-dir outputs --report evaluation_report.html
```

Metrics included:

- MCD
- F0 RMSE
- Speaker Similarity
- CER
- WER
- MOSNet

## Inference

```bash
python inference.py --text "আজকের আবহাওয়া খুব সুন্দর" --speaker SPK_300
```

Supports:

- voice cloning
- speaker selection
- emotion selection
- dialect selection

## Export

```bash
python push_to_hub.py --checkpoint checkpoints/best_model.pt --output-dir exports --repo-id your-org/your-model
```

## Troubleshooting

- Ensure audio files are readable and encoded correctly before preprocessing.
- If the training launcher reports that F5-TTS cannot be imported, add the upstream F5-TTS package or source tree to the environment.
- For GPU memory issues, lower batch size and increase gradient accumulation.
- If speaker embeddings fail, install `speechbrain` and its audio dependencies.

## Testing

```bash
python -m unittest discover -s project/tests
```
