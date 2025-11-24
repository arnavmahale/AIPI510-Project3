"""FastAPI app for sentiment prediction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic import Field

from src.utils.config import load_config, resolve_path


class PredictRequest(BaseModel):
    text: str = Field(..., description="Input text to classify")


class PredictResponse(BaseModel):
    label: int
    probability: float


app = FastAPI(title="SST-2 Sentiment API", version="0.1.0")


def load_model() -> Any:
    cfg = load_config(Path(os.environ.get("CONFIG_PATH", "config/config.yaml")))
    base_dir = Path(__file__).resolve().parents[2]  # project root
    artifacts_cfg: Dict[str, Any] = cfg["artifacts"]
    pipeline_path = resolve_path(artifacts_cfg["pipeline_path"], base_dir)
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {pipeline_path}. Train first.")
    return joblib.load(pipeline_path)


model = load_model()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is required.")
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    proba = model.predict_proba([req.text])[0]
    label = int(proba.argmax())
    prob = float(proba.max())
    return PredictResponse(label=label, probability=prob)
