from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import threading
import time
from typing import Dict, Any
from pathlib import Path

app = FastAPI(title="MLflow Dev API")

jobs: Dict[int, Dict[str, Any]] = {}
jobs_lock = threading.Lock()


class JobRequest(BaseModel):
    script: str = ""  # relative path to script to run


@app.get("/health")
def health():
    return {"status": "ok"}


def _start_process(cmd: list) -> int:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pid = proc.pid
    with jobs_lock:
        jobs[pid] = {
            "pid": pid,
            "cmd": cmd,
            "started_at": time.time(),
            "returncode": None,
            "stdout": None,
            "stderr": None,
        }

    def _wait_and_record(p, pid_):
        out, err = p.communicate()
        with jobs_lock:
            jobs[pid_]["returncode"] = p.returncode
            try:
                jobs[pid_]["stdout"] = out.decode(errors="replace")
            except Exception:
                jobs[pid_]["stdout"] = None
            try:
                jobs[pid_]["stderr"] = err.decode(errors="replace")
            except Exception:
                jobs[pid_]["stderr"] = None

    t = threading.Thread(target=_wait_and_record, args=(proc, pid), daemon=True)
    t.start()
    return pid


@app.post("/jobs/train")
def run_train(req: JobRequest = None):
    # default to baseline experiment if script not provided
    script = req.script if req and req.script.strip() else "src/experiments/baseline_experiment.py"
    path = Path(script)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Script not found: {script}")
    cmd = ["python", script]
    pid = _start_process(cmd)
    return {"pid": pid, "script": script}


@app.post("/jobs/evaluate")
def run_evaluate(req: JobRequest = None):
    script = req.script if req and req.script.strip() else "src/experiments/model_comparison_experiment.py"
    path = Path(script)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Script not found: {script}")
    pid = _start_process(["python", script])
    return {"pid": pid, "script": script}


@app.post("/jobs/predict")
def run_predict(req: JobRequest):
    if not req or not req.script:
        raise HTTPException(status_code=400, detail="Provide script path for prediction")
    script = req.script
    path = Path(script)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Script not found: {script}")
    pid = _start_process(["python", script])
    return {"pid": pid, "script": script}


@app.get("/jobs")
def list_jobs():
    with jobs_lock:
        # return a shallow copy to avoid mutation issues
        return {pid: dict(info) for pid, info in jobs.items()}


@app.get("/jobs/{pid}")
def job_status(pid: int):
    with jobs_lock:
        info = jobs.get(pid)
        if not info:
            raise HTTPException(status_code=404, detail="Job not found")
        return info
"""FastAPI serving application for ML pipeline orchestration."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import json
import traceback

from logger import get_logger
from pipelines.train_pipeline import TrainPipeline
from pipelines.evaluation_pipeline import EvaluationPipeline
from pipelines.predict_pipeline import PredictionPipeline
from utils import get_project_root, save_json, load_json

# Initialize FastAPI app
app = FastAPI(
    title="Energy Consumption ML Pipeline",
    description="API for training, evaluating, and predicting energy consumption",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = get_logger("serving_api")
project_root = get_project_root()
status_file = project_root / "artifacts" / "status.json"
DATA_ROOT = project_root / "artifacts" / "data_ingested"
MODEL_ROOT = project_root / "artifacts" / "models"
MODEL_CHOICES = ["linear_regression", "random_forest", "xgboost", "lightgbm", "ensemble"]
FEATURE_SETS = ["all", "seasonal", "lag_only", "rolling_only", "seasonal_plus_lag"]
TUNING_METHODS = ["grid", "random"]
TUNING_GRID_TYPES = ["coarse", "fine"]

# Models
class TrainRequest(BaseModel):
    model_name: str = "random_forest"
    training_data_file: Optional[str] = None
    feature_set: str = "all"
    tune: bool = False
    tuning_method: str = "grid"
    tuning_grid_type: str = "coarse"
    tuning_n_iter: int = 20

class EvaluateRequest(BaseModel):
    validation_data_file: Optional[str] = None
    model_file: Optional[str] = None
    feature_set: str = "all"

class PredictRequest(BaseModel):
    input_data_file: str
    output_file: Optional[str] = None
    model_file: Optional[str] = None
    feature_set: str = "all"

class EvaluateRequest(BaseModel):
    validation_data_file: Optional[str] = None
    model_file: Optional[str] = None

class PredictRequest(BaseModel):
    input_data_file: str
    output_file: Optional[str] = None
    model_file: Optional[str] = None

class StatusResponse(BaseModel):
    last_training_status: str
    last_training_metrics: Optional[Dict] = None
    last_evaluation_status: str
    last_evaluation_metrics: Optional[Dict] = None
    model_path: Optional[str] = None
    timestamp: str


def get_status() -> Dict:
    """Load current status from file."""
    if status_file.exists():
        return load_json(status_file)
    return {
        "last_training_status": "no_runs",
        "last_training_metrics": None,
        "last_evaluation_status": "no_runs",
        "last_evaluation_metrics": None,
        "model_path": None,
        "timestamp": ""
    }


def list_data_files(section: str) -> list:
    directory = DATA_ROOT / section
    if not directory.exists():
        return []
    return sorted([str(path.relative_to(project_root).as_posix()) for path in directory.glob("*.csv")])


def list_model_files() -> list:
    if not MODEL_ROOT.exists():
        return []
    return sorted([str(path.relative_to(project_root).as_posix()) for path in MODEL_ROOT.glob("*.pkl")])


@app.get("/api/options", tags=["System"])
async def get_pipeline_options():
    """Get available data and model options for the UI."""
    try:
        return {
            "training_files": list_data_files("training"),
            "validation_files": list_data_files("validation"),
            "model_files": list_model_files(),
            "model_choices": MODEL_CHOICES,
            "feature_sets": FEATURE_SETS,
            "tuning_methods": TUNING_METHODS,
            "tuning_grid_types": TUNING_GRID_TYPES,
        }
    except Exception as exc:
        logger.error(f"Failed to get options: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


def update_status(key: str, value, metrics: Optional[Dict] = None):
    """Update status file."""
    status = get_status()
    status[key] = value
    if metrics:
        if key == "last_training_status":
            status["last_training_metrics"] = metrics
        elif key == "last_evaluation_status":
            status["last_evaluation_metrics"] = metrics
        else:
            status[f"{key}_metrics"] = metrics
    from datetime import datetime
    status["timestamp"] = datetime.now().isoformat()
    save_json(status, status_file)


# API Endpoints
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Energy Consumption ML Pipeline"}


@app.get("/api/status", tags=["Status"])
async def get_pipeline_status():
    """Get current pipeline status and metrics."""
    try:
        status = get_status()
        return status
    except Exception as exc:
        logger.error(f"Failed to get status: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/train", tags=["Pipeline"])
async def train_model(request: TrainRequest):
    """Train a new model."""
    logger.info(f"Training request received: {request.model_name}")
    try:
        train_pipeline = TrainPipeline(experiment_name="api_training")
        result = train_pipeline.run(
            model_name=request.model_name,
            training_data_file=request.training_data_file,
            feature_set=request.feature_set,
            tune=request.tune,
            tuning_method=request.tuning_method,
            tuning_grid_type=request.tuning_grid_type,
            tuning_n_iter=request.tuning_n_iter,
            tags={"source": "api"}
        )
        
        update_status(
            "last_training_status",
            "success",
            metrics=result.get("metrics")
        )
        status = get_status()
        status["model_path"] = result.get("local_model_path")
        save_json(status, status_file)
        
        logger.info(f"Training completed: {result}")
        return {
            "status": "success",
            "message": f"Model {request.model_name} trained successfully",
            "result": result
        }
    except Exception as exc:
        logger.error(f"Training failed: {exc}\n{traceback.format_exc()}")
        update_status("last_training_status", f"failed: {str(exc)}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/evaluate", tags=["Pipeline"])
async def evaluate_model(request: EvaluateRequest):
    """Evaluate a trained model."""
    logger.info("Evaluation request received")
    try:
        eval_pipeline = EvaluationPipeline(experiment_name="api_evaluation")
        result = eval_pipeline.run(
            local_model_path=request.model_file,
            validation_data_file=request.validation_data_file,
            feature_set=request.feature_set,
            tags={"source": "api"}
        )
        
        update_status(
            "last_evaluation_status",
            "success",
            metrics=result.get("metrics")
        )
        
        logger.info(f"Evaluation completed: {result}")
        return {
            "status": "success",
            "message": "Model evaluation completed",
            "result": result
        }
    except Exception as exc:
        logger.error(f"Evaluation failed: {exc}\n{traceback.format_exc()}")
        update_status("last_evaluation_status", f"failed: {str(exc)}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/predict", tags=["Pipeline"])
async def predict(request: PredictRequest):
    """Generate predictions using a trained model."""
    logger.info(f"Prediction request received for {request.input_data_file}")
    try:
        predict_pipeline = PredictionPipeline()
        predictions = predict_pipeline.run(
            input_path=request.input_data_file,
            output_path=request.output_file,
            local_model_path=request.model_file,
            feature_set=request.feature_set,
        )
        
        logger.info(f"Predictions generated: shape {predictions.shape}")
        return {
            "status": "success",
            "message": f"Predictions generated for {len(predictions)} samples",
            "predictions_count": len(predictions),
            "output_file": str(request.output_file) if request.output_file else None
        }
    except Exception as exc:
        logger.error(f"Prediction failed: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def get_frontend():
    """Serve the frontend UI."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Energy Consumption ML Pipeline</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 28px;
            margin-bottom: 10px;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 14px;
        }
        
        .content {
            padding: 30px;
        }
        
        .status-box {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 5px;
        }
        
        .status-box h2 {
            font-size: 16px;
            margin-bottom: 15px;
            color: #333;
        }
        
        .status-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #e0e0e0;
            font-size: 14px;
        }
        
        .status-item:last-child {
            border-bottom: none;
        }
        
        .status-label {
            font-weight: 600;
            color: #555;
        }
        
        .status-value {
            color: #888;
        }
        
        .status-success {
            color: #28a745;
            font-weight: 600;
        }
        
        .status-failed {
            color: #dc3545;
            font-weight: 600;
        }
        
        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .control-box {
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 8px;
            background: #f9f9f9;
            transition: all 0.3s ease;
        }
        
        .control-box:hover {
            border-color: #667eea;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
        }
        
        .control-box h3 {
            font-size: 16px;
            margin-bottom: 15px;
            color: #333;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        .form-group label {
            display: block;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 5px;
            color: #555;
            text-transform: uppercase;
        }
        
        .form-group input,
        .form-group select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            font-family: inherit;
        }
        
        .form-group input:focus,
        .form-group select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .btn {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 4px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        .btn-primary:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-secondary {
            background: #6c757d;
            color: white;
        }
        
        .btn-secondary:hover {
            background: #5a6268;
        }
        
        .response-box {
            background: #f8f9fa;
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
            margin-top: 15px;
            font-size: 13px;
            max-height: 300px;
            overflow-y: auto;
            display: none;
        }
        
        .response-box.show {
            display: block;
        }
        
        .response-box.success {
            border-left: 4px solid #28a745;
            background: #f0f9f6;
        }
        
        .response-box.error {
            border-left: 4px solid #dc3545;
            background: #faf6f7;
        }
        
        .spinner {
            display: inline-block;
            width: 14px;
            height: 14px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .loading {
            display: none;
            font-size: 13px;
            color: #667eea;
            font-weight: 600;
        }
        
        .loading.active {
            display: inline-block;
        }
        
        .metrics-box {
            background: #e7f3ff;
            border-left: 4px solid #0066cc;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            font-size: 13px;
        }
        
        .metrics-box strong {
            color: #0066cc;
        }
        
        .footer {
            background: #f8f9fa;
            padding: 15px 30px;
            border-top: 1px solid #ddd;
            font-size: 12px;
            color: #888;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ Energy Consumption ML Pipeline</h1>
            <p>Train, Evaluate, and Predict Energy Consumption Forecasts</p>
        </div>
        
        <div class="content">
            <!-- Status Section -->
            <div class="status-box" id="statusBox">
                <h2>📊 Pipeline Status</h2>
                <div class="status-item">
                    <span class="status-label">Last Training:</span>
                    <span class="status-value" id="trainingStatus">Loading...</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Training Metrics:</span>
                    <span class="status-value" id="trainingMetrics">—</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Last Evaluation:</span>
                    <span class="status-value" id="evaluationStatus">Loading...</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Evaluation Metrics:</span>
                    <span class="status-value" id="evaluationMetrics">—</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Model Path:</span>
                    <span class="status-value" id="modelPath">—</span>
                </div>
            </div>
            
            <!-- Controls Section -->
            <div class="controls">
                <!-- Train Control -->
                <div class="control-box">
                    <h3>🎯 Train Model</h3>
                    <div class="form-group">
                        <label>Training Data</label>
                        <select id="trainDataFile"></select>
                    </div>
                    <div class="form-group">
                        <label>Model</label>
                        <select id="trainModelName"></select>
                    </div>
                    <div class="form-group">
                        <label>Feature Set</label>
                        <select id="trainFeatureSet"></select>
                    </div>
                    <div class="form-group">
                        <label>
                            <input type="checkbox" id="trainTuneToggle" onchange="toggleTrainAdvanced()" />
                            Enable hyperparameter tuning
                        </label>
                    </div>
                    <div id="trainAdvancedOptions" style="display:none; margin-top: 10px;">
                        <div class="form-group">
                            <label>Tuning Method</label>
                            <select id="trainTuningMethod"></select>
                        </div>
                        <div class="form-group">
                            <label>Grid Type</label>
                            <select id="trainTuningGridType"></select>
                        </div>
                        <div class="form-group">
                            <label>Iterations (random search)</label>
                            <input type="number" id="trainTuningNIter" value="20" min="1" step="1" />
                        </div>
                    </div>
                    <button class="btn btn-primary" onclick="trainModel()">Start Training</button>
                    <div class="loading" id="trainLoading"><span class="spinner"></span> Training...</div>
                    <div class="response-box" id="trainResponse"></div>
                </div>
                
                <!-- Evaluate Control -->
                <div class="control-box">
                    <h3>✅ Evaluate Model</h3>
                    <div class="form-group">
                        <label>Validation Data</label>
                        <select id="evalValidationFile"></select>
                    </div>
                    <div class="form-group">
                        <label>Model File</label>
                        <select id="evalModelFile"></select>
                    </div>
                    <div class="form-group">
                        <label>Feature Set</label>
                        <select id="evalFeatureSet"></select>
                    </div>
                    <button class="btn btn-primary" onclick="evaluateModel()">Start Evaluation</button>
                    <div class="loading" id="evalLoading"><span class="spinner"></span> Evaluating...</div>
                    <div class="response-box" id="evalResponse"></div>
                </div>
                
                <!-- Predict Control -->
                <div class="control-box">
                    <h3>🔮 Make Predictions</h3>
                    <div class="form-group">
                        <label>Input Data File</label>
                        <select id="predictInputFile"></select>
                    </div>
                    <div class="form-group">
                        <label>Model File</label>
                        <select id="predictModelFile"></select>
                    </div>
                    <div class="form-group">
                        <label>Feature Set</label>
                        <select id="predictFeatureSet"></select>
                    </div>
                    <div class="form-group">
                        <label>Output File (optional)</label>
                        <input type="text" id="predictOutputFile" placeholder="Path to save predictions">
                    </div>
                    <button class="btn btn-primary" onclick="predictModel()">Generate Predictions</button>
                    <div class="loading" id="predictLoading"><span class="spinner"></span> Predicting...</div>
                    <div class="response-box" id="predictResponse"></div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            Energy Consumption ML Pipeline | API v1.0 | MLflow Integration
        </div>
    </div>
    
    <script>
        // Load status and UI options on page load
        window.addEventListener('load', () => {
            loadStatus();
            loadOptions();
            // Refresh status every 5 seconds
            setInterval(loadStatus, 5000);
        });
        
        async function populateSelect(selectId, items, defaultLabel = null) {
            const select = document.getElementById(selectId);
            select.innerHTML = '';
            if (defaultLabel) {
                const option = document.createElement('option');
                option.value = '';
                option.textContent = defaultLabel;
                select.appendChild(option);
            }

            for (const item of items) {
                const option = document.createElement('option');
                option.value = item;
                option.textContent = item;
                select.appendChild(option);
            }
        }

        async function loadOptions() {
            try {
                const response = await fetch('/api/options');
                const data = await response.json();

                await populateSelect('trainDataFile', data.training_files, 'Select training file');
                await populateSelect('trainModelName', data.model_choices);
                await populateSelect('trainFeatureSet', data.feature_sets);
                await populateSelect('trainTuningMethod', data.tuning_methods);
                await populateSelect('trainTuningGridType', data.tuning_grid_types);

                await populateSelect('evalValidationFile', data.validation_files, 'Select validation file');
                await populateSelect('evalModelFile', data.model_files, 'Use latest model');
                await populateSelect('evalFeatureSet', data.feature_sets);

                await populateSelect('predictInputFile', data.validation_files, 'Select input file');
                await populateSelect('predictModelFile', data.model_files, 'Use latest model');
                await populateSelect('predictFeatureSet', data.feature_sets);
                document.getElementById('trainModelName').value = data.model_choices.includes('random_forest') ? 'random_forest' : data.model_choices[0];
                document.getElementById('trainFeatureSet').value = 'all';
                document.getElementById('evalFeatureSet').value = 'all';
                document.getElementById('predictFeatureSet').value = 'all';
                document.getElementById('trainTuningMethod').value = 'grid';
                document.getElementById('trainTuningGridType').value = 'coarse';
            } catch (error) {
                console.error('Error loading options:', error);
            }
        }

        function toggleTrainAdvanced() {
            const advanced = document.getElementById('trainAdvancedOptions');
            advanced.style.display = document.getElementById('trainTuneToggle').checked ? 'block' : 'none';
        }

        async function loadStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                document.getElementById('trainingStatus').textContent = data.last_training_status || 'No runs';
                document.getElementById('evaluationStatus').textContent = data.last_evaluation_status || 'No runs';
                
                if (data.last_training_metrics) {
                    const metrics = data.last_training_metrics;
                    document.getElementById('trainingMetrics').textContent = 
                        `RMSE: ${metrics.rmse?.toFixed(3)}, MAE: ${metrics.mae?.toFixed(3)}`;
                } else {
                    document.getElementById('trainingMetrics').textContent = '—';
                }
                
                if (data.last_evaluation_metrics) {
                    const metrics = data.last_evaluation_metrics;
                    document.getElementById('evaluationMetrics').textContent = 
                        `RMSE: ${metrics.rmse?.toFixed(3)}, MAE: ${metrics.mae?.toFixed(3)}`;
                } else {
                    document.getElementById('evaluationMetrics').textContent = '—';
                }
                
                if (data.model_path) {
                    document.getElementById('modelPath').textContent = data.model_path.split('/').pop();
                } else {
                    document.getElementById('modelPath').textContent = '—';
                }
            } catch (error) {
                console.error('Error loading status:', error);
            }
        }
        
        async function trainModel() {
            const modelName = document.getElementById('trainModelName').value;
            const trainingDataFile = document.getElementById('trainDataFile').value || null;
            const featureSet = document.getElementById('trainFeatureSet').value;
            const tune = document.getElementById('trainTuneToggle').checked;
            const tuningMethod = document.getElementById('trainTuningMethod').value;
            const tuningGridType = document.getElementById('trainTuningGridType').value;
            const tuningNIter = parseInt(document.getElementById('trainTuningNIter').value, 10) || 20;
            const loading = document.getElementById('trainLoading');
            const responseBox = document.getElementById('trainResponse');

            loading.classList.add('active');
            responseBox.classList.remove('show', 'success', 'error');

            try {
                const response = await fetch('/api/train', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model_name: modelName,
                        training_data_file: trainingDataFile,
                        feature_set: featureSet,
                        tune: tune,
                        tuning_method: tuningMethod,
                        tuning_grid_type: tuningGridType,
                        tuning_n_iter: tuningNIter
                    })
                });

                const data = await response.json();

                if (response.ok) {
                    responseBox.classList.add('show', 'success');
                    responseBox.innerHTML = '<strong>✓ Training completed successfully!</strong><br>' +
                        JSON.stringify(data.result, null, 2);
                } else {
                    responseBox.classList.add('show', 'error');
                    responseBox.innerHTML = '<strong>✗ Training failed!</strong><br>' + data.detail;
                }

                loadStatus();
                loadOptions();
            } catch (error) {
                responseBox.classList.add('show', 'error');
                responseBox.innerHTML = '<strong>✗ Error:</strong> ' + error.message;
            } finally {
                loading.classList.remove('active');
            }
        }
        
        async function evaluateModel() {
            const modelFile = document.getElementById('evalModelFile').value || null;
            const validationDataFile = document.getElementById('evalValidationFile').value || null;
            const featureSet = document.getElementById('evalFeatureSet').value;
            const loading = document.getElementById('evalLoading');
            const responseBox = document.getElementById('evalResponse');

            loading.classList.add('active');
            responseBox.classList.remove('show', 'success', 'error');

            try {
                const response = await fetch('/api/evaluate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model_file: modelFile,
                        validation_data_file: validationDataFile,
                        feature_set: featureSet
                    })
                });

                const data = await response.json();

                if (response.ok) {
                    responseBox.classList.add('show', 'success');
                    responseBox.innerHTML = '<strong>✓ Evaluation completed!</strong><br>' +
                        JSON.stringify(data.result, null, 2);
                } else {
                    responseBox.classList.add('show', 'error');
                    responseBox.innerHTML = '<strong>✗ Evaluation failed!</strong><br>' + data.detail;
                }

                loadStatus();
            } catch (error) {
                responseBox.classList.add('show', 'error');
                responseBox.innerHTML = '<strong>✗ Error:</strong> ' + error.message;
            } finally {
                loading.classList.remove('active');
            }
        }
        
        async function predictModel() {
            const inputFile = document.getElementById('predictInputFile').value;
            const modelFile = document.getElementById('predictModelFile').value || null;
            const featureSet = document.getElementById('predictFeatureSet').value;
            const outputFile = document.getElementById('predictOutputFile').value || null;
            const loading = document.getElementById('predictLoading');
            const responseBox = document.getElementById('predictResponse');

            if (!inputFile) {
                responseBox.classList.add('show', 'error');
                responseBox.innerHTML = '<strong>✗ Error:</strong> Input file is required';
                return;
            }

            loading.classList.add('active');
            responseBox.classList.remove('show', 'success', 'error');

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        input_data_file: inputFile,
                        output_file: outputFile,
                        model_file: modelFile,
                        feature_set: featureSet
                    })
                });

                const data = await response.json();

                if (response.ok) {
                    responseBox.classList.add('show', 'success');
                    responseBox.innerHTML = '<strong>✓ Predictions generated!</strong><br>' +
                        JSON.stringify(data, null, 2);
                } else {
                    responseBox.classList.add('show', 'error');
                    responseBox.innerHTML = '<strong>✗ Prediction failed!</strong><br>' + data.detail;
                }
            } catch (error) {
                responseBox.classList.add('show', 'error');
                responseBox.innerHTML = '<strong>✗ Error:</strong> ' + error.message;
            } finally {
                loading.classList.remove('active');
            }
        }
    </script>
</body>
</html>
    """


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
