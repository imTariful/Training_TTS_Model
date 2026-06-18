from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(slots=True)
class F5TTSModelConfig:
    model_name_or_path: str | None = None
    speaker_embedding_dim: int = 192
    emotion_embedding_dim: int = 64
    dialect_embedding_dim: int = 64


class F5TTSAdapter:
    def __init__(self, config: F5TTSModelConfig | None = None) -> None:
        self.config = config or F5TTSModelConfig()
        self.model = self._build_model()

    def _build_model(self) -> torch.nn.Module:
        try:
            from f5_tts.model import F5TTS  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "F5-TTS is not installed. Install the upstream F5-TTS package or provide a compatible adapter."
            ) from exc
        if self.config.model_name_or_path:
            return F5TTS.from_pretrained(self.config.model_name_or_path)
        return F5TTS()

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.model(**batch)

    def generate(self, **kwargs: Any) -> Any:
        if hasattr(self.model, "generate"):
            return self.model.generate(**kwargs)
        raise RuntimeError("The underlying F5-TTS model does not expose generate().")

    def save_pretrained(self, path: str | Path) -> None:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(str(output))
        else:
            torch.save(self.model.state_dict(), output / "pytorch_model.bin")
