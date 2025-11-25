# AIPI510-Project3 — SST-2 Sentiment Analysis Pipeline

End-to-end sentiment analysis system using the SST-2 dataset with reproducible training, MLflow tracking, containerized FastAPI service, Google Cloud deployment, and a Streamlit front-end interface.

## Project Overview and Goals

This project implements a complete machine learning pipeline for binary sentiment classification on movie reviews. The system demonstrates:

- **Data Engineering**: Automated data ingestion from Hugging Face datasets with cloud storage on Google Cloud Storage (GCS)
- **Model Development**: TF-IDF feature extraction with logistic regression, including hyperparameter configuration
- **Experiment Tracking**: MLflow integration for reproducible model training and evaluation metrics
- **API Service**: RESTful FastAPI endpoint for real-time sentiment predictions
- **Containerization**: Docker-based deployment for portability and scalability
- **Cloud Deployment**: Production deployment on Google Cloud Run with automatic scaling
- **User Interface**: Interactive Streamlit web application for model testing

The goal is to provide a production-ready sentiment analysis service that can classify text as positive or negative sentiment with high accuracy.

## Dataset Description

**Stanford Sentiment Treebank (SST-2)**

- **Source**: [Hugging Face GLUE benchmark](https://huggingface.co/datasets/glue) (`glue/sst2`)
- **Description**: Binary sentiment classification dataset containing movie review sentences
- **Labels**:
  - `1` = Positive sentiment
  - `0` = Negative sentiment
- **Size**:
  - Training set: ~67,000 samples
  - Validation set: ~800 samples
  - Test set: ~1,800 samples
- **Cloud Storage**: Preprocessed datasets are stored in Google Cloud Storage bucket `gs://sst2-sentiment-m3/`
  - `sst2_train.csv`
  - `sst2_val.csv`
  - `sst2_test.csv`

## Model Architecture and Evaluation

### Architecture

The model uses a scikit-learn pipeline with two stages:

1. **TF-IDF Vectorizer**
   - `max_features`: 20,000
   - `ngram_range`: (1, 2) — captures unigrams and bigrams
   - `lowercase`: True
   - Feature extraction converts text into numerical TF-IDF weighted vectors

2. **Logistic Regression Classifier**
   - `C`: 1.0 (inverse regularization strength)
   - `max_iter`: 500
   - `solver`: default (lbfgs)
   - Binary classification with probability estimates

### Evaluation Metrics

- **Primary Metric**: Accuracy
- **Secondary Metric**: F1 Score (macro)
- **Validation Performance**: Tracked and logged via MLflow
- All hyperparameters and metrics are logged to MLflow for experiment tracking and reproducibility

### Training Pipeline

The training process ([src/train.py](src/train.py)):
1. Loads configuration from [config.yaml](config.yaml)
2. Fetches preprocessed data from GCS
3. Trains TF-IDF + Logistic Regression pipeline
4. Evaluates on validation set
5. Logs parameters and metrics to MLflow
6. Saves trained pipeline to `artifacts/model_pipeline.joblib`

## Cloud Services Used

### Google Cloud Storage (GCS)
- **Purpose**: Dataset storage and versioning
- **Bucket**: `gs://sst2-sentiment-m3/`
- **Files**: Preprocessed train/validation/test CSV files
- **Access**: Public read access via `gcsfs` library

### Google Cloud Run
- **Purpose**: Serverless API deployment
- **Service**: `sst2-api`
- **Region**: `us-east1`
- **Features**:
  - Automatic scaling based on request volume
  - HTTPS endpoint with SSL
  - Containerized FastAPI application
  - No authentication required (public access)
- **Deployed API**: https://sst2-api-664742743732.us-east1.run.app/predict

### Streamlit Cloud
- **Purpose**: Front-end web application hosting
- **Integration**: Calls deployed Cloud Run API endpoint
- **Features**:
  - Interactive text input interface
  - Real-time sentiment predictions
  - Probability scores displayed
- **Deployed App**: https://aipi510-project3-sentiment.streamlit.app/

### MLflow
- **Purpose**: Experiment tracking and model versioning
- **Storage**: Local filesystem (`mlruns/` directory)
- **Tracked Items**:
  - Hyperparameters (TF-IDF settings, model params)
  - Evaluation metrics (accuracy, F1 score)
  - Trained model artifacts

## Setup and Usage Instructions

### Prerequisites
- Python 3.10 or 3.11
- Docker (for containerization)
- Google Cloud SDK (for deployment)
- Git

### Local Development Setup

1. **Clone the repository and navigate to project directory**:
   ```bash
   cd AIPI510-Project3
   ```

2. **Create and activate virtual environment**:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model**:
   ```bash
   python -m src.train
   ```
   This will:
   - Download data from GCS
   - Train the TF-IDF + Logistic Regression pipeline
   - Log metrics to MLflow
   - Save model to `artifacts/model_pipeline.joblib`

5. **View MLflow tracking**:
   ```bash
   mlflow ui
   ```
   Open http://localhost:5000 to view experiments

### Running the API Locally

1. **Start the FastAPI server**:
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Test the API**:
   ```bash
   curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "I loved this movie!"}'
   ```

3. **View API documentation**: Open http://localhost:8000/docs

### Running with Docker

1. **Build the Docker image**:
   ```bash
   docker build -t sentiment-api .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8000:8000 sentiment-api
   ```

### Running the Streamlit Frontend Locally

1. **Install frontend dependencies**:
   ```bash
   pip install -r requirements_frontend.txt
   ```

2. **Run Streamlit app** (pointing to deployed API):
   ```bash
   streamlit run app.py
   ```

3. **Access the app**: Open http://localhost:8501

### Deploying to Google Cloud Run

1. **Authenticate with Google Cloud**:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Build and push Docker image**:
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/sentiment-api
   ```

3. **Deploy to Cloud Run**:
   ```bash
   gcloud run deploy sst2-api \
     --image gcr.io/YOUR_PROJECT_ID/sentiment-api \
     --platform managed \
     --region us-east1 \
     --allow-unauthenticated \
     --port 8000
   ```

4. **Get the service URL**: The deployment will output the HTTPS endpoint

### Configuration

All pipeline settings are configured in [config.yaml](config.yaml):

- **Data paths**: GCS bucket URLs for train/val/test data
- **Feature extraction**: TF-IDF parameters (max_features, ngram_range)
- **Model hyperparameters**: Logistic regression settings (C, max_iter)
- **MLflow tracking**: Experiment name and tracking URI
- **Artifacts**: Model save path

## Live Deployment Links

- **API Endpoint**: https://sst2-api-664742743732.us-east1.run.app/predict
- **API Documentation**: https://sst2-api-664742743732.us-east1.run.app/docs
- **Frontend Application**: https://aipi510-project3-sentiment.streamlit.app/

### Using the Deployed API

**Example cURL request**:
```bash
curl -X POST "https://sst2-api-664742743732.us-east1.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was amazing!"}'
```

**Example response**:
```json
{
  "label": 1,
  "probability": 0.9234
}
```

## Project Structure

```
AIPI510-Project3/
├── config.yaml              # Pipeline configuration
├── requirements.txt         # Python dependencies
├── requirements_frontend.txt # Streamlit dependencies
├── Dockerfile              # Container definition for API
├── app.py                  # Streamlit frontend application
├── src/
│   ├── __init__.py
│   ├── data_ingest.py      # Data loading from GCS
│   ├── train.py            # Training pipeline with MLflow
│   └── api/
│       └── main.py         # FastAPI application
├── artifacts/              # Saved model artifacts (gitignored)
│   └── model_pipeline.joblib
└── mlruns/                 # MLflow tracking data (gitignored)
```

## Technologies and Libraries

- **ML/Data**: scikit-learn, pandas, numpy, joblib
- **Experiment Tracking**: MLflow
- **API Framework**: FastAPI, Uvicorn, Pydantic
- **Frontend**: Streamlit, requests
- **Cloud**: gcsfs (GCS access), Google Cloud Run
- **Containerization**: Docker
- **Configuration**: PyYAML

## Future Improvements

- Add test dataset evaluation metrics
- Implement model versioning and A/B testing
- Add authentication to API endpoints
- Expand to multi-class sentiment (neutral, very positive, very negative)
- Integrate with CI/CD pipeline for automated deployments
- Add comprehensive unit and integration tests

## Attribution

Portions of this codebase were developed with assistance from ChatGPT and Claude Code.

## License

This project is for educational purposes as part of Duke AIPI 510 coursework.
