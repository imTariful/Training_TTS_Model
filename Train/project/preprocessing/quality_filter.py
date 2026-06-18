from __future__ import annotations

from dataclasses import dataclass

from ..utils.metadata import MetadataRow


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
class QualityDecision:
    keep: bool
    reasons: list[str]


def filter_sample(row: MetadataRow, thresholds: QualityThresholds) -> QualityDecision:
    reasons: list[str] = []
    if row.get_float("transcription_confidence") < thresholds.transcription_confidence:
        reasons.append("transcription_confidence")
    if row.get_float("audio_quality_score") < thresholds.audio_quality_score:
        reasons.append("audio_quality_score")
    if row.get_float("speaker_verification_score") < thresholds.speaker_verification_score:
        reasons.append("speaker_verification_score")
    if row.get_float("snr") < thresholds.snr:
        reasons.append("snr")
    if row.get_float("cer") > thresholds.cer:
        reasons.append("cer")
    duration = row.get_float("duration")
    if duration < thresholds.min_duration:
        reasons.append("duration_too_short")
    if duration > thresholds.max_duration:
        reasons.append("duration_too_long")
    return QualityDecision(keep=not reasons, reasons=reasons)
