from __future__ import annotations

from dataclasses import dataclass

import os


@dataclass(slots=True)
class GpuProfile:
    name: str
    vram_gb: float
    batch_size: int
    gradient_accumulation: int
    num_workers: int
    mixed_precision: bool


GPU_PROFILES = [
    GpuProfile("RTX 3060", 12.0, 4, 4, 4, True),
    GpuProfile("RTX 4060", 8.0, 2, 8, 4, True),
    GpuProfile("RTX 4070", 12.0, 4, 4, 6, True),
    GpuProfile("RTX 4090", 24.0, 8, 2, 8, True),
    GpuProfile("A100", 40.0, 16, 1, 8, True),
]


def detect_gpu_name() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    index = torch.cuda.current_device()
    return torch.cuda.get_device_name(index)


def detect_vram_gb() -> float:
    try:
        import torch
    except Exception:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    return float(total) / (1024 ** 3)


def recommend_training_settings(vram_gb: float | None = None) -> GpuProfile:
    vram = vram_gb if vram_gb is not None else detect_vram_gb()
    if vram >= 32:
        return GPU_PROFILES[-1]
    if vram >= 20:
        return GPU_PROFILES[3]
    if vram >= 10:
        return GPU_PROFILES[2]
    if vram >= 8:
        return GPU_PROFILES[1]
    return GPU_PROFILES[0]


def get_cuda_env() -> dict[str, str]:
    env = dict(os.environ)
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    return env
