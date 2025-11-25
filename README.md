# AIPI510-Project3 — SST-2 Sentiment Pipeline

End-to-end sentiment analysis system using the SST-2 dataset: reproducible training, MLflow tracking, containerized FastAPI service, Cloud Run deployment, and a Streamlit front-end.

## Stack

- Data/ML: Hugging Face `datasets`, scikit-learn (TF-IDF + logistic regression), MLflow for metrics/artifacts.
- API: FastAPI + Uvicorn; Dockerized for Cloud Run.
- Front-end: Streamlit app calling the deployed `/predict` endpoint.
- Requirements: `requirements_backend.txt` (API/training), `requirements.txt` (Streamlit frontend)

## Repo layout

- `config.yaml` — pipeline, model, and API configuration.
- `src/data_ingest.py` — data loading from GCS.
- `src/train.py` — training pipeline with MLflow logging.
- `src/api/main.py` — FastAPI app loading the trained pipeline and exposing `/predict`.
- `app.py` — Streamlit UI that calls the API.
- `requirements_backend.txt` — backend dependencies (API, training, MLflow).
- `requirements.txt` — frontend dependencies (Streamlit).
- `Dockerfile` — container definition for API deployment.
- `artifacts/` — saved model pipeline (ignored in git).
- `mlruns/` — MLflow local tracking (ignored in git).

## Getting started

1. Python 3.10+ recommended. Create env and install deps:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements_backend.txt
   ```
2. Train and log to MLflow; saves model pipeline to `artifacts/`:
   ```bash
   python -m src.train
   ```
   This downloads data from GCS (`gs://sst2-sentiment-m3/`), trains the model, and saves artifacts.
3. Run API locally:
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   # then POST to /predict with {"text": "I loved this movie!"}
   ```
4. Docker (local):
   ```bash
   docker build -t sentiment-api .
   docker run -p 8000:8000 sentiment-api
   ```
5. Streamlit UI (local):
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Config

Tune paths, split ratios, TF-IDF settings, and model params in `config.yaml`. MLflow tracking URI is set to a local `mlruns/` directory by default.

## API

- `POST /predict` — body: `{"text": "string"}`; returns label and probability.

## Deployment targets

- API: Google Cloud Run (containerized FastAPI).
- Front-end: Streamlit Cloud (consumes live API).
- Data hosting: push `data/processed/*.csv` to a public location (GitHub raw or Hugging Face dataset) and set `data.remote_base_url` for reproducible downloads.

## Front-end (Streamlit)

- Local: `streamlit run app.py` (update API URL in `app.py` if needed).
- Deploy to Streamlit Cloud:
  1. Push this repo to GitHub.
  2. Create a new Streamlit app pointing to `app.py`.
  3. Update the API URL in `app.py` to your deployed Cloud Run endpoint.

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

- **SST-2 (Stanford Sentiment Treebank)**: https://huggingface.co/datasets/stanfordnlp/sst2
- Binary labels: 1 = positive, 0 = negative. Metric: accuracy (also report macro F1).
- Cleaned CSVs stored in Google Cloud Storage bucket `gs://sst2-sentiment-m3/` for reproducible access.

## Model

- TF-IDF + Logistic Regression (scikit-learn).
- Hyperparameters in `config/config.yaml` (n-grams, max_features, C, max_iter).
- Training/eval logged to MLflow at `mlruns/` (local filesystem URI).

## Notes

- Train before building/deploying the API so the model artifact exists at `artifacts/model_pipeline.joblib`.
- To run the API elsewhere, ship the artifact with the image or mount it; otherwise, the service will fail to start.

## Deliverables checklist

- Code for preprocessing, training, serving, deployment scripts.
- `config.yaml` for pipeline params.
- `requirements_backend.txt` (API/training) and `requirements.txt` (frontend).
- Dockerfile for API.
- MLflow logging.
- FastAPI `/predict` endpoint.
- Front-end (Streamlit) that consumes the deployed API.
- README with overview, dataset info, model summary, cloud services, setup, and links.

## Live Deployment

### Deployed Links

- **API Endpoint**: https://sst2-api-664742743732.us-east1.run.app/predict
- **API Documentation**: https://sst2-api-664742743732.us-east1.run.app/docs
- **Frontend Application**: https://aipi510-project3-sentiment.streamlit.app/

### Cloud Services

- **Google Cloud Storage**: Dataset hosting (`gs://sst2-sentiment-m3/`)
- **Google Cloud Run**: Serverless API deployment with automatic scaling
- **Streamlit Cloud**: Frontend web application hosting

### Example API Usage

```bash
curl -X POST "https://sst2-api-664742743732.us-east1.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was amazing!"}'
```

**Response**:

```json
{
  "label": 1,
  "probability": 0.9234
}
```

## Attribution

- Portions of the codebase were written with assistance from ChatGPT.
