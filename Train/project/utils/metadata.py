from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import csv

REQUIRED_COLUMNS = {
    "utterance_id",
    "text",
    "normalized_text",
    "lang",
    "speaker_id",
    "audio_path",
    "audio",
    "duration",
    "gender",
    "age_group",
    "dialect",
    "scenario",
    "speaking_style",
    "emotion",
    "utterance_pitch_mean",
    "utterance_pitch_std",
    "snr",
    "speaking_rate",
    "cer",
    "c50",
    "recording_device",
    "recording_environment",
    "transcription_confidence",
    "audio_quality_score",
    "phoneme_sequence",
    "language_mix_ratio",
    "speaker_verification_score",
    "district",
    "qualification",
    "occupation",
    "split",
}


@dataclass(slots=True)
class MetadataRow:
    values: dict[str, str]

    def get_float(self, key: str, default: float = 0.0) -> float:
        value = self.values.get(key, "")
        if value == "":
            return default
        try:
            return float(value)
        except ValueError:
            return default

    def get_int(self, key: str, default: int = 0) -> int:
        value = self.values.get(key, "")
        if value == "":
            return default
        try:
            return int(float(value))
        except ValueError:
            return default

    def get_str(self, key: str, default: str = "") -> str:
        return self.values.get(key, default)


def read_metadata_csv(path: str | Path) -> list[MetadataRow]:
    """Read entire metadata CSV into memory (for small datasets)."""
    return list(iter_metadata_csv(path))


def iter_metadata_csv(path: str | Path) -> Iterator[MetadataRow]:
    """Stream metadata CSV row by row (for large datasets)."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield MetadataRow(values={key: (value or "").strip() for key, value in row.items()})


def resolve_path(raw_path: str, root: str | Path | None = None) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute() or root is None:
        return candidate
    return Path(root) / candidate


def write_metadata_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
