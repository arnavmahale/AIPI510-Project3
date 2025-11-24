"""Train a sentiment classifier on SST-2 and log to MLflow."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from src.utils.config import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sentiment model.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file.",
    )
    return parser.parse_args()


def load_splits(cfg: Dict[str, Any], base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_cfg = cfg["data"]
    train_df = pd.read_csv(resolve_path(data_cfg["cleaned_train_path"], base_dir))
    val_df = pd.read_csv(resolve_path(data_cfg["cleaned_val_path"], base_dir))
    test_df = pd.read_csv(resolve_path(data_cfg["cleaned_test_path"], base_dir))
    return train_df, val_df, test_df


def build_pipeline(cfg: Dict[str, Any]) -> Pipeline:
    features = cfg["features"]
    model_cfg = cfg["model"]
    training_cfg = cfg.get("training", {})

    vectorizer = TfidfVectorizer(
        max_features=features.get("max_features"),
        ngram_range=tuple(features.get("ngram_range", (1, 1))),
        lowercase=features.get("lowercase", True),
        stop_words=features.get("stop_words"),
    )

    if model_cfg.get("type") != "logreg":
        raise ValueError("Only 'logreg' model type is supported in this script.")

    clf = LogisticRegression(
        C=model_cfg.get("c", 1.0),
        max_iter=model_cfg.get("max_iter", 500),
        class_weight=model_cfg.get("class_weight"),
        n_jobs=training_cfg.get("n_jobs", 1),
        random_state=model_cfg.get("random_state", 42),
    )

    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def eval_split(model: Pipeline, X, y) -> Dict[str, float]:
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}


def main(args: argparse.Namespace) -> None:
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    base_dir = cfg_path.parent.parent  # project root

    train_df, val_df, test_df = load_splits(cfg, base_dir)
    text_col = cfg["data"]["text_column"]
    target_col = cfg["data"]["target_column"]

    X_train, y_train = train_df[text_col], train_df[target_col]
    X_val, y_val = val_df[text_col], val_df[target_col]
    X_test, y_test = test_df[text_col], test_df[target_col]

    pipeline = build_pipeline(cfg)

    mlflow_cfg = cfg["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    artifacts_cfg = cfg["artifacts"]
    artifacts_dir = resolve_path(artifacts_cfg["dir"], base_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = resolve_path(artifacts_cfg["pipeline_path"], base_dir)
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="logreg-tfidf") as run:
        mlflow.log_params(
            {
                "max_features": cfg["features"].get("max_features"),
                "ngram_range": cfg["features"].get("ngram_range"),
                "lowercase": cfg["features"].get("lowercase", True),
                "stop_words": cfg["features"].get("stop_words"),
                "C": cfg["model"].get("c", 1.0),
                "max_iter": cfg["model"].get("max_iter", 500),
                "class_weight": cfg["model"].get("class_weight"),
                "random_state": cfg["model"].get("random_state", 42),
                "n_jobs": cfg.get("training", {}).get("n_jobs", -1),
            }
        )

        print("Training model...")
        pipeline.fit(X_train, y_train)

        print("Evaluating on validation and test splits...")
        val_metrics = eval_split(pipeline, X_val, y_val)
        test_metrics = eval_split(pipeline, X_test, y_test)

        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        joblib.dump(pipeline, pipeline_path)
        mlflow.log_artifact(pipeline_path)

        # Save metrics summary to a text file for quick inspection.
        summary_path = artifacts_dir / "metrics_summary.txt"
        with summary_path.open("w") as f:
            f.write("Validation metrics:\n")
            for k, v in val_metrics.items():
                f.write(f"{k}: {v:.4f}\n")
            f.write("\nTest metrics:\n")
            for k, v in test_metrics.items():
                f.write(f"{k}: {v:.4f}\n")
        mlflow.log_artifact(summary_path)

        print(f"Run ID: {run.info.run_id}")
        print(f"Saved pipeline to: {pipeline_path}")
        print(f"Val metrics: {val_metrics}")
        print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main(parse_args())
