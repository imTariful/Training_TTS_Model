from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..preprocessing.normalize_bn import normalize_bangla_text
from ..preprocessing.phonemizer_bn import phonemize_bangla
from ..training.model_adapter import F5TTSAdapter, F5TTSModelConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class InferenceRequest:
    text: str
    speaker_id: str | None = None
    emotion: str | None = None
    dialect: str | None = None
    reference_audio: str | None = None
    speaker_embeddings_path: str | None = None


class TTSInferenceEngine:
    def __init__(self, model: F5TTSAdapter | None = None) -> None:
        self.model = model or F5TTSAdapter(F5TTSModelConfig())

    def synthesize(self, request: InferenceRequest) -> Any:
        normalized_text = normalize_bangla_text(request.text)
        phonemes = phonemize_bangla(normalized_text)
        conditioning = {
            "text": normalized_text,
            "phonemes": phonemes,
            "speaker_id": request.speaker_id,
            "emotion": request.emotion,
            "dialect": request.dialect,
            "reference_audio": request.reference_audio,
        }
        return self.model.generate(**conditioning)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bangla F5-TTS inference")
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker", dest="speaker_id", default=None)
    parser.add_argument("--emotion", default=None)
    parser.add_argument("--dialect", default=None)
    parser.add_argument("--reference-audio", default=None)
    parser.add_argument("--output", default="output.wav")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    engine = TTSInferenceEngine()
    output = engine.synthesize(
        InferenceRequest(
            text=args.text,
            speaker_id=args.speaker_id,
            emotion=args.emotion,
            dialect=args.dialect,
            reference_audio=args.reference_audio,
        )
    )
    if isinstance(output, torch.Tensor):
        torch.save(output, args.output)
    elif hasattr(output, "save"):
        output.save(args.output)
    else:
        Path(args.output).write_text(str(output), encoding="utf-8")
    LOGGER.info("Saved inference output to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
