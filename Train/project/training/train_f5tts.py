from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..configs.defaults import ProjectConfig
from ..utils.gpu import detect_gpu_name, detect_vram_gb, recommend_training_settings

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class LaunchConfig:
    project_name: str
    project_dir: Path
    exp_name: str
    learning_rate: float
    batch_size_per_gpu: int
    batch_size_type: str
    max_samples: int
    grad_accumulation_steps: int
    max_grad_norm: float
    epochs: int
    num_warmup_updates: int
    save_per_updates: int
    keep_last_n_checkpoints: int
    last_per_updates: int
    finetune: bool
    pretrain: str | None
    tokenizer: str
    tokenizer_path: str | None
    mixed_precision: str
    logger_backend: str | None
    bnb_optimizer: bool
    stream: bool
    dry_run: bool


class F5TTSLauncher:
    def __init__(self, config: ProjectConfig | None = None) -> None:
        self.config = config or ProjectConfig().resolve()

    def build_command(self, launch: LaunchConfig) -> list[str]:
        finetune_cli = self._resolve_finetune_cli()
        command = [
            self._resolve_accelerate(),
            "launch",
        ]
        if launch.mixed_precision and launch.mixed_precision != "none":
            command.append(f"--mixed_precision={launch.mixed_precision}")
        command.extend(
            [
                str(finetune_cli),
                "--exp_name",
                launch.exp_name,
                "--dataset_name",
                launch.project_name,
                "--learning_rate",
                str(launch.learning_rate),
                "--batch_size_per_gpu",
                str(launch.batch_size_per_gpu),
                "--batch_size_type",
                launch.batch_size_type,
                "--max_samples",
                str(launch.max_samples),
                "--grad_accumulation_steps",
                str(launch.grad_accumulation_steps),
                "--max_grad_norm",
                str(launch.max_grad_norm),
                "--epochs",
                str(launch.epochs),
                "--num_warmup_updates",
                str(launch.num_warmup_updates),
                "--save_per_updates",
                str(launch.save_per_updates),
                "--keep_last_n_checkpoints",
                str(launch.keep_last_n_checkpoints),
                "--last_per_updates",
                str(launch.last_per_updates),
                "--tokenizer",
                launch.tokenizer,
            ]
        )
        if launch.finetune:
            command.append("--finetune")
        if launch.pretrain:
            command.extend(["--pretrain", launch.pretrain])
        if launch.tokenizer_path:
            command.extend(["--tokenizer_path", launch.tokenizer_path])
        if launch.logger_backend:
            command.extend(["--logger", launch.logger_backend])
        if launch.bnb_optimizer:
            command.append("--bnb_optimizer")
        command.append("--log_samples")
        return command

    def launch(self, launch: LaunchConfig) -> int:
        self._validate_project_dir(launch.project_dir)
        command = self.build_command(launch)
        LOGGER.info("GPU: %s (%.1f GiB)", detect_gpu_name(), detect_vram_gb())
        LOGGER.info("Command: %s", " ".join(command))
        if launch.dry_run:
            return 0

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("CUDA_VISIBLE_DEVICES", env.get("CUDA_VISIBLE_DEVICES", "0"))
        process = subprocess.Popen(
            command,
            cwd=str(self.config.project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
        return process.wait()

    def recommend_settings(self) -> dict[str, int]:
        profile = recommend_training_settings()
        return {
            "batch_size_per_gpu": profile.batch_size,
            "grad_accumulation_steps": profile.gradient_accumulation,
            "num_workers": profile.num_workers,
        }

    def _resolve_accelerate(self) -> str:
        accelerate = shutil.which("accelerate")
        if accelerate:
            return accelerate
        raise RuntimeError("accelerate is not available on PATH. Install accelerate and run accelerate config first.")

    def _resolve_finetune_cli(self) -> Path:
        candidates = [
            self.config.project_root / "src" / "f5_tts" / "train" / "finetune_cli.py",
            Path(sys.prefix) / "Lib" / "site-packages" / "f5_tts" / "train" / "finetune_cli.py",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        try:
            import f5_tts.train.finetune_cli as finetune_cli  # type: ignore

            return Path(finetune_cli.__file__).resolve()
        except Exception as exc:
            raise RuntimeError(
                "Could not find f5_tts.train.finetune_cli.py. Install F5-TTS in editable mode or place its source tree in this workspace."
            ) from exc

    def _validate_project_dir(self, project_dir: Path) -> None:
        required = [project_dir / "raw.arrow", project_dir / "duration.json", project_dir / "vocab.txt"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Project directory is missing required F5-TTS files: {missing}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch official F5-TTS fine-tuning")
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--exp-name", default="F5TTS_v1_Base")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size-per-gpu", type=int, default=None)
    parser.add_argument("--batch-size-type", choices=["frame", "sample"], default="frame")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--grad-accumulation-steps", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-warmup-updates", type=int, default=100)
    parser.add_argument("--save-per-updates", type=int, default=500)
    parser.add_argument("--keep-last-n-checkpoints", type=int, default=-1)
    parser.add_argument("--last-per-updates", type=int, default=100)
    parser.add_argument("--finetune", action="store_true", default=True)
    parser.add_argument("--no-finetune", action="store_false", dest="finetune")
    parser.add_argument("--pretrain", default=None)
    parser.add_argument("--tokenizer", choices=["pinyin", "char", "custom"], default="custom")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--mixed-precision", choices=["none", "fp16", "bf16"], default="fp16")
    parser.add_argument("--logger", choices=["wandb", "tensorboard", "none"], default="tensorboard")
    parser.add_argument("--bnb-optimizer", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ProjectConfig().resolve()
    project_dir = Path(args.project_dir) if args.project_dir else config.dataset.root_dir / args.project_name
    tokenizer_path = args.tokenizer_path
    if args.tokenizer == "custom" and not tokenizer_path:
        tokenizer_path = str(project_dir / "vocab.txt")
    launcher = F5TTSLauncher(config)
    recommendations = launcher.recommend_settings()
    launch = LaunchConfig(
        project_name=args.project_name,
        project_dir=project_dir,
        exp_name=args.exp_name,
        learning_rate=args.learning_rate,
        batch_size_per_gpu=args.batch_size_per_gpu or recommendations["batch_size_per_gpu"],
        batch_size_type=args.batch_size_type,
        max_samples=args.max_samples,
        grad_accumulation_steps=args.grad_accumulation_steps or recommendations["grad_accumulation_steps"],
        max_grad_norm=args.max_grad_norm,
        epochs=args.epochs,
        num_warmup_updates=args.num_warmup_updates,
        save_per_updates=args.save_per_updates,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        last_per_updates=args.last_per_updates,
        finetune=args.finetune,
        pretrain=args.pretrain,
        tokenizer=args.tokenizer,
        tokenizer_path=tokenizer_path,
        mixed_precision=args.mixed_precision,
        logger_backend=None if args.logger == "none" else args.logger,
        bnb_optimizer=args.bnb_optimizer,
        stream=True,
        dry_run=args.dry_run,
    )
    exit_code = launcher.launch(launch)
    return exit_code


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
