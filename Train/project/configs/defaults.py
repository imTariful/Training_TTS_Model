from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int = 24000
    target_channels: int = 1
    min_duration: float = 1.0
    max_duration: float = 30.0
    silence_trim_db: float = 30.0
    loudness_target_lufs: float = -16.0
    max_peak_dbfs: float = -1.0
    lowpass_hz: int = 12000


@dataclass(slots=True)
class QualityThresholds:
    transcription_confidence: float = 0.95
    audio_quality_score: float = 0.80
    speaker_verification_score: float = 0.90
    snr: float = 20.0
    cer: float = 0.05
    min_duration: float = 1.0
    max_duration: float = 30.0


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    gradient_accumulation: int = 1
    weight_decay: float = 0.01
    num_workers: int = 4
    mixed_precision: bool = True
    deepspeed_config: str | None = None
    resume_from: str | None = None
    tensorboard_dir: str = "runs"
    wandb_project: str | None = None


@dataclass(slots=True)
class DatasetConfig:
    root_dir: Path = Path("data")
    processed_audio_dir: Path = Path("processed_audio")
    metadata_file: Path = Path("metadata.csv")
    train_file: Path = Path("train.txt")
    valid_file: Path = Path("valid.txt")
    valid_ratio: float = 0.02
    random_seed: int = 42


@dataclass(slots=True)
class ProjectConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    quality: QualityThresholds = field(default_factory=QualityThresholds)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    project_root: Path = Path(".")
    hub_model_id: str | None = None

    def resolve(self) -> "ProjectConfig":
        self.project_root = self.project_root.resolve()
        self.dataset.root_dir = (self.project_root / self.dataset.root_dir).resolve()
        self.dataset.processed_audio_dir = (self.project_root / self.dataset.processed_audio_dir).resolve()
        self.dataset.metadata_file = (self.project_root / self.dataset.metadata_file).resolve()
        self.dataset.train_file = (self.project_root / self.dataset.train_file).resolve()
        self.dataset.valid_file = (self.project_root / self.dataset.valid_file).resolve()
        if self.training.tensorboard_dir:
            self.training.tensorboard_dir = str((self.project_root / self.training.tensorboard_dir).resolve())
        return self


def load_config(overrides: dict[str, Any] | None = None) -> ProjectConfig:
    config = ProjectConfig().resolve()
    if not overrides:
        return config
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
