# Data Directory Structure

This directory contains all audio and metadata for your F5-TTS training pipeline.

## Structure

```
data/
├── raw_audio/          # Original, unprocessed audio files
├── processed_audio/    # Processed audio (24kHz mono WAV, trimmed, normalized)
├── metadata.csv        # Main metadata file (audio paths, transcripts, speaker info)
└── <project_name>/     # Prepared F5-TTS project (generated automatically)
```

## Metadata CSV Format

Your `metadata.csv` should have these columns (you can add more for extra information):

| Column | Required | Description |
|--------|----------|-------------|
| `utterance_id` | Yes | Unique ID for each audio clip |
| `audio_path` | Yes | Path to audio file (relative to `data/`) |
| `text` | Yes | Transcript of the audio |
| `normalized_text` | No | Pre-normalized transcript (optional) |
| `speaker_id` | Yes | Unique speaker identifier |
| `duration` | No | Audio duration in seconds (optional, will be calculated automatically) |
| `gender` | No | Speaker gender (optional) |
| `age_group` | No | Speaker age group (optional) |
| `dialect` | No | Speaker dialect (optional) |
| `emotion` | No | Emotion in audio (optional) |
| `split` | No | Train/valid split (optional, will be split automatically if not provided) |

## Example Metadata CSV

```csv
utterance_id,audio_path,text,speaker_id
bn_001,raw_audio/bn_001.wav,আজকের আবহাওয়া খুব সুন্দর,SPK_001
bn_002,raw_audio/bn_002.wav,আমি ভাত খাই,SPK_001
bn_003,raw_audio/bn_003.wav,সূর্য উঠছে,SPK_002
```
