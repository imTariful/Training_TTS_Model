from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..configs.defaults import AudioConfig
from ..utils.audio import AudioMetrics, calculate_audio_metrics, load_audio, save_audio, to_mono, resample_audio
from ..utils.logging import setup_logging
from ..utils.metadata import MetadataRow, read_metadata_csv, resolve_path, write_metadata_csv

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PreprocessResult:
    utterance_id: str
    source_path: str
    processed_path: str
    duration: float
    clipped: bool
    peak: float
    rms: float


class AudioPreprocessor:
    def __init__(self, config: AudioConfig, input_root: str | Path, output_root: str | Path) -> None:
        self.config = config
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)

    def preprocess_row(self, row: MetadataRow) -> PreprocessResult | None:
        source_path = self._resolve_audio_path(row)
        if source_path is None or not source_path.exists():
            return None

        samples, sample_rate = load_audio(source_path)
        samples = self._trim_silence(samples)
        samples = to_mono(samples)
        samples = self._noise_filter(samples, sample_rate)
        samples = self._normalize_loudness(samples)
        samples = np.clip(samples, -1.0, 1.0)
        samples = resample_audio(samples[:, None], sample_rate, self.config.sample_rate)
        samples = to_mono(samples)

        output_path = self.output_root / f"{row.get_str('utterance_id')}.wav"
        save_audio(output_path, samples, self.config.sample_rate)
        metrics = calculate_audio_metrics(samples[:, None] if samples.ndim == 1 else samples, self.config.sample_rate)
        return PreprocessResult(
            utterance_id=row.get_str("utterance_id"),
            source_path=str(source_path),
            processed_path=str(output_path),
            duration=metrics.duration,
            clipped=metrics.clipped,
            peak=metrics.peak,
            rms=metrics.rms,
        )

    def preprocess_dataset(self, rows: list[MetadataRow]) -> list[PreprocessResult]:
        results: list[PreprocessResult] = []
        for row in rows:
            try:
                result = self.preprocess_row(row)
            except Exception as exc:
                LOGGER.exception("Failed to preprocess %s: %s", row.get_str("utterance_id"), exc)
                continue
            if result is not None:
                results.append(result)
        return results

    def _resolve_audio_path(self, row: MetadataRow) -> Path | None:
        raw_path = row.get_str("audio_path") or row.get_str("audio")
        return resolve_path(raw_path, self.input_root)

    def _trim_silence(self, samples: np.ndarray) -> np.ndarray:
        mono = to_mono(samples)
        if mono.size == 0:
            return samples
        threshold = 10 ** (-self.config.silence_trim_db / 20.0)
        indices = np.flatnonzero(np.abs(mono) > threshold)
        if indices.size == 0:
            return samples
        start = max(int(indices[0]) - int(0.05 * len(mono)), 0)
        end = min(int(indices[-1]) + int(0.05 * len(mono)), len(mono))
        trimmed = mono[start:end]
        return trimmed[:, None]

    def _normalize_loudness(self, samples: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak <= 0.0:
            return samples
        target_peak = 10 ** (self.config.max_peak_dbfs / 20.0)
        return samples * (target_peak / peak)

    def _noise_filter(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        try:
            from scipy.signal import butter, filtfilt
        except Exception:
            return samples
        nyquist = sample_rate / 2.0
        cutoff = min(self.config.lowpass_hz, int(nyquist * 0.95))
        if cutoff <= 0:
            return samples
        b, a = butter(4, cutoff / nyquist, btype="low")
        return filtfilt(b, a, samples)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess audio for F5-TTS training")
    parser.add_argument("--metadata-file", default="metadata.csv")
    parser.add_argument("--input-root", default=".")
    parser.add_argument("--output-root", default="processed_audio")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--silence-trim-db", type=float, default=30.0)
    parser.add_argument("--max-peak-dbfs", type=float, default=-1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    rows = read_metadata_csv(args.metadata_file)
    config = AudioConfig(sample_rate=args.sample_rate, silence_trim_db=args.silence_trim_db, max_peak_dbfs=args.max_peak_dbfs)
    preprocessor = AudioPreprocessor(config, args.input_root, args.output_root)
    results = preprocessor.preprocess_dataset(rows)
    manifest_rows = [
        {
            "utterance_id": result.utterance_id,
            "source_path": result.source_path,
            "processed_path": result.processed_path,
            "duration": result.duration,
            "clipped": result.clipped,
            "peak": result.peak,
            "rms": result.rms,
        }
        for result in results
    ]
    write_metadata_csv(Path(args.output_root) / "preprocessing_manifest.csv", manifest_rows, list(manifest_rows[0].keys()) if manifest_rows else ["utterance_id", "source_path", "processed_path", "duration", "clipped", "peak", "rms"])
    LOGGER.info("Preprocessed %d utterances", len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
