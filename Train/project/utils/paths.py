from __future__ import annotations

from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def relative_to_root(path: str | Path, root: str | Path) -> Path:
    return Path(path).resolve().relative_to(Path(root).resolve())
