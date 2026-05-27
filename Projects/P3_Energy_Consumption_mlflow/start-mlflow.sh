#!/usr/bin/env bash
# Start local MLflow server against local Postgres and MinIO for development

export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:-minioadmin}
export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:-minioadmin}
export MLFLOW_S3_ENDPOINT_URL=${MLFLOW_S3_ENDPOINT_URL:-http://localhost:9000}

# Backend and artifact locations (adjust if needed)
BACKEND_STORE_URI=${BACKEND_STORE_URI:-postgresql://mlflow:mlflow@localhost:5432/mlflow}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-s3://mlflow-artifacts/}

echo "Starting MLflow server"
mlflow server \
  --backend-store-uri "$BACKEND_STORE_URI" \
  --default-artifact-root "$ARTIFACT_ROOT" \
  --host 0.0.0.0 --port 5000
