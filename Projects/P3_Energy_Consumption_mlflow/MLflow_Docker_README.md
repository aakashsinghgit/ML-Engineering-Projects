# Local Docker Quickstart (MLflow + FastAPI Serving)

This setup orchestrates the entire Energy Consumption project stack via Docker Compose. It provisions:
- A local **MLflow Tracking Server** backed by Postgres (metadata) and MinIO (S3-compatible artifact storage).
- A **FastAPI Serving UI** to interact with your machine learning pipelines.

Prerequisites:
- Docker and Docker Compose installed locally.

Quick steps:

1. Build and start all services:

```bash
docker compose up -d --build
```

2. Open the Interfaces:
- **Serving UI (FastAPI):** http://localhost:8000
- **MLflow Tracking UI:** http://localhost:5000

3. Alternatively, you can run MLflow locally (outside containers) after starting Postgres and MinIO with compose, using `start-mlflow.sh`:

```bash
# Ensure Postgres and MinIO are running via docker compose
./start-mlflow.sh
```

Notes:
- Default MinIO credentials: `minioadmin` / `minioadmin` (for local testing only). You can access the MinIO console at http://localhost:9001.
- Default Postgres credentials: `mlflow` / `mlflow` (database `mlflow`).
- For production, use a managed Postgres and a managed S3 bucket; update env variables or the MLflow server command accordingly.
