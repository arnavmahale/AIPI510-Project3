"""Download, clean, and split the SST-2 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SST-2 data.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def resolve_path(p: str, base: Path) -> Path:
    candidate = Path(p)
    return candidate if candidate.is_absolute() else base / candidate


def clean_text(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    df = df[[text_col, label_col]].copy()
    df[text_col] = df[text_col].fillna("").str.strip()
    df = df[df[text_col] != ""]
    df[label_col] = df[label_col].astype(int)
    return df.reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    label_col: str,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strat = df[label_col] if stratify else None
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )
    val_ratio = val_size / (1 - test_size)
    strat_train = train_val_df[label_col] if stratify else None
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        random_state=random_state,
        stratify=strat_train,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def main(args: argparse.Namespace) -> None:
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    base_dir = cfg_path.parent.parent  # project root (assumes config/config.yaml)

    data_cfg = cfg["data"]
    split_cfg = cfg["split"]

    source = data_cfg["source"]
    subset = data_cfg.get("subset")
    print(f"Loading dataset {source} {subset or ''} from Hugging Face...")
    if subset:
        hf_ds = load_dataset(source, subset)
    else:
        hf_ds = load_dataset(source)

    # Use train + validation splits (GLUE SST-2 has unlabeled test).
    train_df_raw = hf_ds["train"].to_pandas()
    val_df_raw = hf_ds["validation"].to_pandas()
    combined_df = pd.concat([train_df_raw, val_df_raw], ignore_index=True)

    text_col = data_cfg["text_column"]
    target_col = data_cfg["target_column"]
    combined_df = clean_text(combined_df, text_col, target_col)

    train_df, val_df, test_df = split_data(
        combined_df,
        label_col=target_col,
        test_size=split_cfg["test_size"],
        val_size=split_cfg["val_size"],
        random_state=split_cfg["random_state"],
        stratify=split_cfg.get("stratify", True),
    )

    out_paths = {
        "train": resolve_path(data_cfg["cleaned_train_path"], base_dir),
        "val": resolve_path(data_cfg["cleaned_val_path"], base_dir),
        "test": resolve_path(data_cfg["cleaned_test_path"], base_dir),
    }
    for p in out_paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_paths["train"], index=False)
    val_df.to_csv(out_paths["val"], index=False)
    test_df.to_csv(out_paths["test"], index=False)

    print("Saved splits:")
    print(f"  train: {out_paths['train']} ({len(train_df)})")
    print(f"  val:   {out_paths['val']} ({len(val_df)})")
    print(f"  test:  {out_paths['test']} ({len(test_df)})")


if __name__ == "__main__":
    main(parse_args())
