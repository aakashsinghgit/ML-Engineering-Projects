# Local MLflow (Docker) Quickstart

This sets up a local MLflow server backed by Postgres and MinIO (S3-compatible) for artifact storage.

Prerequisites:
- Docker and Docker Compose installed locally.

Quick steps:

1. Start services:

```bash
docker compose up -d
```

2. Run the MLflow server inside the `mlflow` service (it starts automatically via `docker-compose` build):

```bash
docker compose up --build mlflow
```

3. Alternatively run MLflow locally (outside containers) after starting Postgres and MinIO with compose, using `start-mlflow.sh`:

```bash
# Ensure Postgres and MinIO are running via docker compose
./start-mlflow.sh
```

4. Open the UI: http://localhost:5000

Notes:
- Default MinIO credentials: `minioadmin` / `minioadmin` (for local testing only).
- Default Postgres credentials: `mlflow` / `mlflow` (database `mlflow`).
- For production, use a managed Postgres and a managed S3 bucket; update env variables or the MLflow server command accordingly.
