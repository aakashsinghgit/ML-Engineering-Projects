# ML Engineering Projects

Production-oriented ML Engineering projects — from model training and serving to MLOps pipelines, Kubernetes deployments, and monitoring systems. Every project is built with real-world engineering practices: modular code, CI/CD, Docker, experiment tracking, and proper serving layers.

**Background:** 5 years of Azure cloud + CI/CD engineering applied to building and shipping ML systems. These aren't tutorial notebooks — they're engineered end-to-end.

***

## 📁 Repository Structure

Each project lives in its own subfolder with independent code, data, and documentation.

```
ML-Engineering-Projects/
├── Projects/
│   ├── P1-loan-predictor/
│   ├── P2-health-insurance-predictor/
│   ├── P3-health-insurance-mlflow-tracker/
│   ├── P4-fraud-detection-api/
│   ├── P5-churn-prediction-pipeline/
│   ├── P6-house-price-cicd/
│   ├── P7-movie-recommender-kubernetes/
│   ├── P8-taxi-demand-monitoring/
│   ├── P9-ecommerce-recommender/
│   └── P10-ml-platform-capstone/
└── README.md
```

***

## 🗺️ Engineering Progressions

Projects are structured in five phases — each phase adds a new layer of production engineering on top of the last.

***

## Phase 1 — Model Training & Serving Foundations

### P1 · Loan Predictor App ✅

A binary classification app that predicts loan approval status, served via a REST API.

**Engineering highlights:**
- Trained and serialised a scikit-learn classifier
- Served predictions through a Flask REST endpoint
- Pickle-based model loading at inference time

**Stack:** Python · scikit-learn · Flask · Pickle

***

### P2 · Health Insurance Price Predictor ✅

Regression model predicting health insurance premiums, built with a production-grade modular structure.

**Engineering highlights:**
- Modular project layout: components, pipelines, logging, exception handling
- CI/CD pipeline with GitHub Actions — automated test and build on every push
- Dockerised for reproducible, environment-independent deployment
- OpenAPI documentation via Swagger UI

**Stack:** Python · scikit-learn · FastAPI · Docker · GitHub Actions

***

## Phase 2 — Experiment Tracking & Pipelines

### P3 · Health Insurance MLflow Tracker 🔨

Adds a full experiment tracking and model registry layer on top of the P2 Health Insurance dataset — turning ad-hoc training runs into a governed, versioned ML workflow.

**Engineering highlights:**
- MLflow experiment tracking: parameters, metrics, and artifacts logged per run
- Model registry with Staging → Production → Archived lifecycle transitions
- Compare 5+ model variants (Ridge, Lasso, Random Forest, XGBoost, LightGBM) in a single tracked experiment
- REST API that always loads the model currently in the "Production" stage
- Reproducible training runs via MLflow Projects

**Stack:** Python · scikit-learn · MLflow · FastAPI · Docker

***

### P4 · Fraud Detection API

Real-time fraud detection model served through a high-performance async FastAPI service.

**Engineering highlights:**
- Async FastAPI endpoints with request validation (Pydantic schemas)
- Batch inference endpoint alongside single-record prediction
- Full OpenAPI / Swagger documentation
- Containerised with Docker, ready for cloud deployment
- Imbalanced dataset handling: SMOTE + class weight tuning

**Stack:** Python · scikit-learn · FastAPI · Pydantic · Docker

***

### P5 · Customer Churn Prediction Pipeline

End-to-end data pipeline preparing features for churn prediction, built for reliability and reproducibility.

**Engineering highlights:**
- ETL pipeline: ingestion, schema validation, cleaning, transformation
- Data validation with Great Expectations — catches bad data before it reaches training
- Feature engineering pipeline with scikit-learn custom transformers
- Pipeline serialisation for reuse at inference time

**Stack:** Python · pandas · scikit-learn · Great Expectations · Docker

***

## Phase 3 — Deployment & Orchestration

### P6 · House Price CI/CD Pipeline

Automated retraining and deployment pipeline for a house price prediction model — triggered on every code or data change.

**Engineering highlights:**
- GitHub Actions pipeline: test → train → evaluate → build Docker image → deploy
- Automated model performance gate — deployment blocked if new model underperforms baseline
- Docker image build and push to container registry on merge to main
- Environment promotion: Staging → Production via GitHub environments

**Stack:** Python · scikit-learn · FastAPI · Docker · GitHub Actions

***

### P7 · Movie Recommender on Kubernetes

Collaborative filtering recommendation system deployed to Kubernetes with autoscaling — directly applying Azure VMSS knowledge to ML serving.

**Engineering highlights:**
- Kubernetes deployment with Horizontal Pod Autoscaler (HPA) based on CPU and request latency
- Helm chart for repeatable, configurable deployments
- Liveness and readiness probes for zero-downtime rolling updates
- Load tested with Locust — autoscaling validated under simulated traffic spikes

**Stack:** Python · scikit-learn · FastAPI · Docker · Kubernetes · Helm · Locust

***

## Phase 4 — MLOps & Monitoring

### P8 · Taxi Demand Forecaster with Monitoring

Time series forecasting service with a full production monitoring stack — detecting data drift and model degradation before users notice.

**Engineering highlights:**
- LSTM-based demand forecaster served via FastAPI
- EvidentlyAI drift detection: PSI and KS test on incoming feature distributions
- Prometheus metrics exporter + Grafana dashboards for real-time visibility
- Alerting rules: fires when drift exceeds threshold
- Drift simulation script for testing the monitoring stack end-to-end

**Stack:** Python · PyTorch · FastAPI · EvidentlyAI · Prometheus · Grafana · Docker Compose

***

## Phase 5 — End-to-End Production Systems

### P9 · E-commerce Product Recommender System

Full-stack ML system for product recommendations — data pipeline through serving, with MLflow tracking and CI/CD.

**Engineering highlights:**
- Data pipeline: ingestion → validation → feature engineering → training
- MLflow experiment tracking + model registry with automated promotion
- FastAPI serving endpoint with A/B experiment routing between model versions
- GitHub Actions: automated retraining triggered on data or code change
- Feature store pattern: offline training features + online serving features separated

**Stack:** Python · scikit-learn · MLflow · FastAPI · Docker · GitHub Actions

***

### P10 · ML Platform Capstone

A unified ML platform wiring together every layer built across this repository — the production system a real ML team would operate.

**Engineering highlights:**
- Single Docker Compose stack: MLflow server + model registry + Prometheus + Grafana + Redis feature cache
- Automated pipeline: new data in → retrain → evaluate → promote to registry → deploy → monitor
- Feature store with offline (batch) and online (low-latency) paths
- Full observability: model performance, data drift, and infrastructure metrics in one Grafana dashboard
- Architecture diagram documenting every component and data flow

**Stack:** Python · MLflow · FastAPI · Feast · EvidentlyAI · Prometheus · Grafana · Docker Compose · GitHub Actions

***

## 🛠️ Tech Stack Across This Repository

| Category | Tools |
| :--- | :--- |
| **ML Frameworks** | scikit-learn, PyTorch, XGBoost, LightGBM |
| **Serving** | FastAPI, Flask, Pydantic |
| **Experiment Tracking** | MLflow, Weights & Biases |
| **Containerisation** | Docker, Docker Compose |
| **Orchestration** | Kubernetes, Helm, Azure AKS |
| **CI/CD** | GitHub Actions |
| **Monitoring** | EvidentlyAI, Prometheus, Grafana |
| **Feature Store** | Feast, Redis |
| **Data Validation** | Great Expectations |
| **Load Testing** | Locust |

***

## 🔗 Portfolio

| | Link |
| :--- | :--- |
| 🌐 Portfolio Site | *Coming soon* |
| 💼 LinkedIn | *Add link* |
| 🤖 AI Engineering Projects | *Coming soon* |
| 🧠 Deep Learning Projects | *Coming soon* |
