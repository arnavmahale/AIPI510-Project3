"""Download pre-cleaned SST-2 CSVs from a public URL and save locally."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import requests

from src.utils.config import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download cleaned SST-2 CSVs from remote storage.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file.",
    )
    return parser.parse_args()


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def main(args: argparse.Namespace) -> None:
    cfg_path = Path(args.config).resolve()
    cfg: Dict[str, Any] = load_config(cfg_path)
    base_dir = cfg_path.parent.parent  # project root

    remote_base = cfg["data"].get("remote_base_url", "").rstrip("/")
    if not remote_base:
        raise ValueError("remote_base_url is not set in config/data.")

    files = {
        "train": cfg["data"]["cleaned_train_path"],
        "val": cfg["data"]["cleaned_val_path"],
        "test": cfg["data"]["cleaned_test_path"],
    }

    for split, rel_path in files.items():
        filename = Path(rel_path).name
        url = f"{remote_base}/{filename}"
        dest = resolve_path(rel_path, base_dir)
        print(f"Downloading {split} from {url} -> {dest}")
        download_file(url, dest)
    print("Done.")


if __name__ == "__main__":
    main(parse_args())
