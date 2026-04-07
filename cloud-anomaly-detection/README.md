# Cloud-Native Anomaly Detection

Multivariate time-series anomaly detection using a Transformer autoencoder. Trained locally, tracked with MLflow, containerised with Docker, and deployable to GCP Vertex AI.

## Architecture
```
Multivariate Time-Series → Sliding Window → Patch Embedding
→ Transformer Encoder → Reconstruction Head
→ Reconstruction Error → Anomaly Score → Threshold → Flag
```

## Stack
- **PyTorch** — Transformer encoder model
- **MLflow** — experiment tracking and model registry
- **Docker** — containerisation
- **GCP Vertex AI** — cloud deployment

## Quickstart
```bash
pip install torch mlflow scikit-learn pandas numpy matplotlib
jupyter notebook cloud_anomaly_detection.ipynb
```

## MLflow Tracking
```bash
mlflow ui   # view at http://localhost:5000
```

## Docker Build
```bash
docker build -t anomaly-detector:v1 .
docker run -p 8080:8080 anomaly-detector:v1
```

## Key Features
- Unsupervised reconstruction-based anomaly detection
- Transformer encoder with positional embeddings
- MLflow parameter + metric + model logging
- Percentile-based threshold tuning
- Full Docker + GCP Vertex AI deployment reference
