from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..configs.defaults import ProjectConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ExportArtifacts:
    checkpoint: Path
    safetensors: Path
    hf_directory: Path


class ModelExporter:
    def __init__(self, config: ProjectConfig | None = None) -> None:
        self.config = config or ProjectConfig().resolve()

    def export(self, checkpoint_path: str | Path, output_dir: str | Path = "exports") -> ExportArtifacts:
        checkpoint = Path(checkpoint_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        payload = torch.load(checkpoint, map_location="cpu")
        model_state = payload.get("model_state_dict", payload)
        checkpoint_out = output_path / "checkpoint.pt"
        torch.save(payload, checkpoint_out)
        safetensors_out = output_path / "model.safetensors"
        try:
            from safetensors.torch import save_file

            save_file(model_state, str(safetensors_out))
        except Exception:
            torch.save(model_state, safetensors_out.with_suffix(".bin"))
            safetensors_out = safetensors_out.with_suffix(".bin")
        hf_dir = output_path / "huggingface"
        hf_dir.mkdir(parents=True, exist_ok=True)
        self._write_hf_metadata(hf_dir, payload)
        return ExportArtifacts(checkpoint=checkpoint_out, safetensors=safetensors_out, hf_directory=hf_dir)

    def _write_hf_metadata(self, hf_dir: Path, payload: dict[str, Any]) -> None:
        metadata = {
            "model_type": "f5-tts",
            "framework": "pytorch",
            "config": payload.get("config", {}),
            "experiment": payload.get("experiment", {}),
        }
        (hf_dir / "config.json").write_text(__import__("json").dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    def push(self, artifacts: ExportArtifacts, repo_id: str, token: str | None = None) -> None:
        try:
            from huggingface_hub import HfApi
        except Exception as exc:
            raise RuntimeError("Install huggingface_hub to push exports to the Hugging Face Hub.") from exc
        api = HfApi(token=token)
        api.upload_folder(folder_path=str(artifacts.hf_directory), repo_id=repo_id, repo_type="model")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Export F5-TTS artifacts and optionally push to the Hub")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="exports")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    exporter = ModelExporter()
    artifacts = exporter.export(args.checkpoint, args.output_dir)
    if args.repo_id:
        exporter.push(artifacts, args.repo_id, token=args.token)
    print(str(asdict(artifacts)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
