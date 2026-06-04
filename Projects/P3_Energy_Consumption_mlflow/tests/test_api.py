import sys
from pathlib import Path

# Add src directory to path
src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from serving.api import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "Energy Consumption ML Pipeline"}

@patch('serving.api.list_data_files')
@patch('serving.api.list_model_files')
def test_options(mock_list_model, mock_list_data):
    mock_list_data.return_value = ["data.csv"]
    mock_list_model.return_value = ["model.pkl"]
    
    response = client.get("/api/options")
    assert response.status_code == 200
    data = response.json()
    assert "training_files" in data
    assert "model_choices" in data
    assert "feature_sets" in data

@patch('serving.api.TrainPipeline')
def test_train_model(mock_train_pipeline):
    mock_instance = MagicMock()
    mock_instance.run.return_value = {
        "run_name": "test_run",
        "metrics": {"rmse": 0.1},
        "local_model_path": "model.pkl"
    }
    mock_train_pipeline.return_value = mock_instance

    response = client.post(
        "/api/train",
        json={
            "model_name": "linear_regression",
            "feature_set": "all"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "result" in data
    mock_instance.run.assert_called_once()

@patch('serving.api.EvaluationPipeline')
def test_evaluate_model(mock_eval_pipeline):
    mock_instance = MagicMock()
    mock_instance.run.return_value = {
        "metrics": {"rmse": 0.15}
    }
    mock_eval_pipeline.return_value = mock_instance

    response = client.post(
        "/api/evaluate",
        json={
            "validation_data_file": "data.csv",
            "model_file": "model.pkl",
            "feature_set": "all"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    mock_instance.run.assert_called_once()

@patch('serving.api.PredictionPipeline')
def test_predict_model(mock_predict_pipeline):
    mock_instance = MagicMock()
    # Mocking prediction returning a list/array of 5 predictions
    mock_instance.run.return_value = [1, 2, 3, 4, 5]
    mock_predict_pipeline.return_value = mock_instance

    response = client.post(
        "/api/predict",
        json={
            "input_data_file": "data.csv",
            "model_file": "model.pkl",
            "feature_set": "all"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["predictions_count"] == 5
    mock_instance.run.assert_called_once()

def test_health_response_structure():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "service" in data
    assert len(data.keys()) == 2

def test_train_invalid_model():
    response = client.post(
        "/api/train",
        json={
            "model_name": "invalid_model_type",
            "feature_set": "all"
        }
    )
    assert response.status_code == 500

def test_predict_missing_input():
    response = client.post(
        "/api/predict",
        json={
            "model_file": "model.pkl",
            "feature_set": "all"
        }
    )
    assert response.status_code == 422

def test_frontend_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Energy" in response.text

@patch('serving.api.get_status')
def test_status_no_runs(mock_get_status):
    mock_get_status.return_value = {
        "last_training_status": "no_runs",
        "last_training_metrics": None,
        "last_evaluation_status": "no_runs",
        "last_evaluation_metrics": None,
        "model_path": None,
        "timestamp": ""
    }
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["last_training_status"] == "no_runs"
    assert data["last_evaluation_status"] == "no_runs"
