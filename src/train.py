import joblib
from pathlib import Path

import mlflow
import mlflow.sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from src.data_ingest import load_config, load_clean_data


def main():
    # Load config + data
    cfg = load_config()
    (X_train, y_train), (X_val, y_val), _ = load_clean_data()

    feat_cfg = cfg["features"]
    model_cfg = cfg["model"]
    ml_cfg = cfg["mlflow"]
    artifacts_cfg = cfg["artifacts"]

    # --- Create TF-IDF + Logistic Regression pipeline ---
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=feat_cfg["max_features"],
                    ngram_range=tuple(feat_cfg["ngram_range"]),
                    lowercase=feat_cfg["lowercase"],
                    stop_words=feat_cfg["stop_words"],
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    C=model_cfg["c"],
                    max_iter=model_cfg["max_iter"],
                    class_weight=model_cfg["class_weight"],
                    n_jobs=cfg["training"]["n_jobs"],
                ),
            ),
        ]
    )

    # --- Train ---
    pipeline.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation F1: {f1:.4f}")

    # --- MLflow logging ---
    mlflow.set_tracking_uri(ml_cfg["tracking_uri"])
    mlflow.set_experiment(ml_cfg["experiment_name"])

    with mlflow.start_run():
        # Log params
        mlflow.log_params(
            {
                "max_features": feat_cfg["max_features"],
                "ngram_range": feat_cfg["ngram_range"],
                "C": model_cfg["c"],
                "max_iter": model_cfg["max_iter"],
            }
        )

        # Log metrics
        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("val_f1", f1)

        # Log the whole sklearn pipeline
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

    # --- Save model pipeline locally as joblib ---
    artifacts_dir = Path(artifacts_cfg["dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, artifacts_cfg["pipeline_path"])
    print(f"Saved model pipeline → {artifacts_cfg['pipeline_path']}")


if __name__ == "__main__":
    main()
