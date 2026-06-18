from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import soundfile as sf
except Exception as exc:  # pragma: no cover - optional dependency guard
    sf = None  # type: ignore[assignment]
    _soundfile_error = exc
else:
    _soundfile_error = None


@dataclass(slots=True)
class AudioMetrics:
    duration: float
    sample_rate: int
    channels: int
    peak: float
    rms: float
    clipped: bool


def require_soundfile() -> None:
    if sf is None:
        raise RuntimeError("soundfile is required for audio IO") from _soundfile_error


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    require_soundfile()
    data, sample_rate = sf.read(str(path), always_2d=True)
    return data.astype(np.float32), int(sample_rate)


def save_audio(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    require_soundfile()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), samples, sample_rate)


def to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples
    if samples.shape[1] == 1:
        return samples[:, 0]
    return np.mean(samples, axis=1)


def calculate_audio_metrics(samples: np.ndarray, sample_rate: int) -> AudioMetrics:
    mono = to_mono(samples)
    duration = float(len(mono) / sample_rate) if sample_rate else 0.0
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    clipped = bool(np.any(np.abs(mono) >= 0.999))
    return AudioMetrics(duration=duration, sample_rate=sample_rate, channels=samples.shape[1] if samples.ndim > 1 else 1, peak=peak, rms=rms, clipped=clipped)


def resample_audio(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples
    try:
        import torchaudio.functional as F
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("torchaudio is required for resampling") from exc

    tensor = torch.from_numpy(np.asarray(samples).T)
    resampled = F.resample(tensor, source_rate, target_rate)
    return resampled.T.numpy()
