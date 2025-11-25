from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml


# Root of the project (…/AIPI510-Project3)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> dict:
    """Load YAML config from the project root."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_clean_data():
    """
    Load pre-split SST-2 datasets from GCS based on paths in config.yaml.
    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    cfg = load_config()
    data_cfg = cfg["data"]

    train_path = data_cfg["cleaned_train_path"]
    val_path = data_cfg["cleaned_val_path"]
    test_path = data_cfg["cleaned_test_path"]

    text_col = data_cfg["text_column"]
    target_col = data_cfg["target_column"]

    # pandas + gcsfs can read gs:// URLs directly
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[text_col].astype(str)
    y_train = train_df[target_col]

    X_val = val_df[text_col].astype(str)
    y_val = val_df[target_col]

    X_test = test_df[text_col].astype(str)
    y_test = test_df[target_col]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    # Quick sanity check when you run:  python -m src.data_ingest
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_clean_data()
    print("Train size:", len(X_train))
    print("Val size:", len(X_val))
    print("Test size:", len(X_test))
    print("Sample:", X_train.iloc[0], "→", y_train.iloc[0])
