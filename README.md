# AIPI510-Project3 — SST-2 Sentiment Pipeline

End-to-end sentiment analysis system using the SST-2 dataset: reproducible training, MLflow tracking, containerized FastAPI service, Cloud Run deployment, and a Streamlit front-end.

## Stack
- Data/ML: Hugging Face `datasets`, scikit-learn (TF-IDF + logistic regression), MLflow for metrics/artifacts.
- API: FastAPI + Uvicorn; Dockerized for Cloud Run.
- Front-end: Streamlit app calling the deployed `/predict` endpoint.

## Repo layout
- `config/config.yaml` — pipeline, model, and API configuration.
- `src/data/` — data download/clean scripts (SST-2).
- `src/data/download_processed.py` — helper to fetch cleaned CSVs from a public URL if hosted (GitHub raw/HF).
- `src/pipeline/` — training, evaluation, artifact saving, MLflow logging.
- `src/api/` — FastAPI app loading the trained pipeline and exposing `/predict`.
- `streamlit_app.py` — Streamlit UI that calls the API (`API_URL` env var).
- `requirements.txt` — shared dependencies.
- `data/` — local cached data (ignored in git); cleaned CSVs will be hosted via repo raw link for “cloud storage” requirement.
- `artifacts/` — saved model pipeline (ignored in git).
- `mlruns/` — MLflow local tracking (ignored in git).

## Getting started
1. Python 3.10+ recommended. Create env and install deps:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Prepare data (downloads SST-2 from Hugging Face, cleans, and writes CSVs):
   ```bash
   python -m src.data.prepare_sst2 --config config/config.yaml
   ```
   If you host cleaned CSVs in a public location (e.g., GitHub raw), set `data.remote_base_url` in `config/config.yaml` and download them with:
   ```bash
   python -m src.data.download_processed --config config/config.yaml
   ```
3. Train and log to MLflow; saves model pipeline to `artifacts/`:
   ```bash
   python -m src.pipeline.train --config config/config.yaml
   ```
4. Run API locally:
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   # then POST to /predict with {"text": "I loved this movie!"}
   ```
5. Docker (local):
   ```bash
   docker build -t sentiment-api .
   docker run -p 8000:8000 sentiment-api
   ```
6. Streamlit UI (local):
   ```bash
   API_URL=http://localhost:8000 streamlit run streamlit_app.py
   ```

## Config
Tune paths, split ratios, TF-IDF settings, and model params in `config/config.yaml`. MLflow tracking URI is set to a local `mlruns/` directory by default.

## API
- `POST /predict` — body: `{"text": "string"}`; returns label and probability.

## Deployment targets
- API: Google Cloud Run (containerized FastAPI).
- Front-end: Streamlit Cloud (consumes live API).
- Data hosting: push `data/processed/*.csv` to a public location (GitHub raw or Hugging Face dataset) and set `data.remote_base_url` for reproducible downloads.

## Front-end (Streamlit)
- Local: `API_URL=http://localhost:8000 streamlit run streamlit_app.py`.
- Deploy to Streamlit Cloud:
  1. Push this repo to GitHub.
  2. Create a new Streamlit app pointing to `streamlit_app.py`.
  3. Set secret/environment variable `API_URL` to your deployed Cloud Run endpoint (e.g., `https://<service>-<hash>-uw.a.run.app`).
  4. Add the public Streamlit link to the README.

## Cloud Run deployment (manual steps)
Prereqs: gcloud CLI authenticated to your GCP project, Artifact Registry enabled.
1. Ensure `artifacts/model_pipeline.joblib` exists (run training first).
2. Build and push:
   ```bash
   gcloud builds submit --tag gcr.io/$PROJECT_ID/sentiment-api
   ```
3. Deploy:
   ```bash
   gcloud run deploy sentiment-api \
     --image gcr.io/$PROJECT_ID/sentiment-api \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --port 8000 \
     --set-env-vars CONFIG_PATH=config/config.yaml
   ```
4. Grab the HTTPS URL from Cloud Run and set it as `API_URL` in Streamlit.

## Dataset
- SST-2 (Stanford Sentiment Treebank) via Hugging Face `glue/sst2`.
- Binary labels: 1 = positive, 0 = negative. Metric: accuracy (also report macro F1).
- Cleaned CSVs produced by `src.data.prepare_sst2`. Host `data/processed/*.csv` in a public location (e.g., GitHub raw) and set `data.remote_base_url` for reproducible downloads.

## Model
- TF-IDF + Logistic Regression (scikit-learn).
- Hyperparameters in `config/config.yaml` (n-grams, max_features, C, max_iter).
- Training/eval logged to MLflow at `mlruns/` (local filesystem URI).

## Notes
- Train before building/deploying the API so the model artifact exists at `artifacts/model_pipeline.joblib`.
- To run the API elsewhere, ship the artifact with the image or mount it; otherwise, the service will fail to start.

## Deliverables checklist
- Code for preprocessing, training, serving, deployment scripts.
- `config/config.yaml` for pipeline params.
- `requirements.txt`.
- Dockerfile for API.
- MLflow logging.
- FastAPI `/predict` endpoint.
- Front-end (Streamlit) that consumes the deployed API.
- README with overview, dataset info, model summary, cloud services, setup, and links (fill in Cloud Run + Streamlit URLs when deployed).

## Status
- Scaffolding in place. Next: implement data prep, training pipeline, API, Docker, Cloud Run deploy, and Streamlit UI.
