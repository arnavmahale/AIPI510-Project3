# Attribution
# Portions of the codebase were developed with assistance from ChatGPT and Claude Code.

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

from src.data_ingest import load_config  # reuse same config loader

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load config + get pipeline path from it
cfg = load_config()
PIPELINE_PATH = PROJECT_ROOT / cfg["artifacts"]["pipeline_path"]

pipeline = joblib.load(PIPELINE_PATH)

app = FastAPI(title="SST-2 Sentiment Analysis API")


class PredictRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(input: PredictRequest):
    pred = pipeline.predict([input.text])[0]
    proba = pipeline.predict_proba([input.text])[0][1]
    return {"label": int(pred), "probability": float(proba)}


@app.get("/")
def root():
    return {"message": "SST-2 Sentiment API is running!"}
