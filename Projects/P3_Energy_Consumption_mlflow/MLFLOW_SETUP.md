# MLflow Setup Guide for Energy Consumption Project

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd Projects/P3_Energy_Consumption_mlflow
pip install -r Requirements.txt
```

### 2. Test MLflow Setup
```bash
python src/test_mlflow_setup.py
```

### 3. Run Data Exploration Experiment
```bash
python src/experiment_template.py
```

## 📁 What We've Created

### Core MLflow Files:
- **`src/mlflow_tracking.py`** - Comprehensive MLflow tracking class
- **`src/mlflow_config.py`** - Configuration settings and experiment definitions
- **`src/test_mlflow_setup.py`** - Test script to verify setup
- **`src/experiment_template.py`** - Template for data exploration experiments

## 🔧 MLflow Features Implemented

### 1. **MLflowTracker Class**
- ✅ Experiment creation and management
- ✅ Run tracking with custom names and tags
- ✅ Parameter and metric logging
- ✅ Model logging (sklearn, xgboost, lightgbm, pytorch, tensorflow)
- ✅ Artifact logging (figures, dataframes, files)
- ✅ Data quality validation logging
- ✅ Best run comparison and retrieval

### 2. **Configuration Management**
- ✅ Experiment configurations for different phases
- ✅ Model configurations with default parameters
- ✅ Data split definitions (training: 2007-2008, validation: 2009-2010)
- ✅ Feature engineering configurations
- ✅ Hyperparameter search spaces

### 3. **Experiment Template**
- ✅ Data loading and validation
- ✅ Time series visualization
- ✅ Correlation analysis
- ✅ Data quality assessment
- ✅ MLflow integration for tracking

## 🎯 Next Steps

### Phase 1: Data Exploration (Ready to Run)
1. **Install dependencies**: `pip install -r Requirements.txt`
2. **Test setup**: `python src/test_mlflow_setup.py`
3. **Run exploration**: `python src/experiment_template.py`

### Phase 2: Feature Engineering Experiments
- Create lag features (1, 2, 3, 6, 12, 24, 48, 72 hours)
- Rolling window statistics (mean, std, min, max)
- Seasonal features (hour, day, month, quarter)
- Interaction features

### Phase 3: Model Comparison Experiments
- Linear Regression
- Random Forest
- XGBoost
- LightGBM
- Time series specific models (ARIMA, LSTM)

### Phase 4: Hyperparameter Tuning
- Grid search and random search
- Bayesian optimization
- Cross-validation with time series splits

## 📊 MLflow UI Access

After running experiments, you can view results in the MLflow UI:

```bash
# Start MLflow UI (from project root)
mlflow ui --backend-store-uri file://artifacts/mlflow
```

Then open: http://localhost:5000

## 🔍 Key Benefits

1. **Reproducibility**: Every experiment is logged with exact parameters
2. **Comparison**: Easy comparison of different models and approaches
3. **Artifact Management**: All models, plots, and data samples saved
4. **Collaboration**: Share results with team members
5. **Production Ready**: Easy model promotion to production

## 📝 Usage Examples

### Basic Experiment
```python
from src.mlflow_tracking import MLflowTracker

tracker = MLflowTracker("my_experiment")
with tracker.start_run("test_run"):
    tracker.log_parameters({"param1": "value1"})
    tracker.log_metrics({"accuracy": 0.95})
    tracker.log_model(model, "my_model")
```

### Using the Template
```python
from src.experiment_template import EnergyConsumptionExperiment

experiment = EnergyConsumptionExperiment("data_exploration")
experiment.run_experiment("exploration_run")
```

## 🎉 Ready for Experimentation!

Your MLflow setup is complete and ready for comprehensive ML experimentation. The system will automatically:
- Track all experiments
- Log parameters and metrics
- Save models and artifacts
- Provide comparison tools
- Enable reproducible research

Start with data exploration, then move to feature engineering and model comparison!
