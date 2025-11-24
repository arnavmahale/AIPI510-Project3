from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: Path) -> Dict[str, Any]:
    with Path(path).open("r") as f:
        return yaml.safe_load(f)


def resolve_path(p: str, base: Path) -> Path:
    candidate = Path(p)
    return candidate if candidate.is_absolute() else base / candidate
