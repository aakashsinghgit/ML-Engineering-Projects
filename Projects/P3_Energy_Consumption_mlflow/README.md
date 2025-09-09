# Energy Consumption MLflow Tracker

Welcome to the **Energy Consumption MLflow Tracker** project!  
This project demonstrates how to forecast household energy consumption using time series data, while tracking experiments and model versions with MLflow.

---

## 📈 Project Overview

**Goal:**  
Build a machine learning pipeline to predict household energy consumption and manage experiments using MLflow.

**Key Features:**
- Time series regression for energy usage forecasting.
- Modular code structure with clear separation of data, features, and models.
- MLflow integration for tracking parameters, metrics, and artifacts.
- Model registry for version control and reproducibility.

---

## 🗂️ Project Structure

```
P3-energy-consumption-mlflow/
├── data/                # Raw and processed datasets
├── notebooks/           # Exploratory data analysis and prototyping
├── src/                 # Source code (data, features, models, pipelines)
├── mlruns/              # MLflow tracking directory
├── requirements.txt     # Project dependencies
├── README.md            # Project documentation
└── ...
```

---

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aakashsinghgit/ML-Engineering-Projects.git
   cd ML-Engineering-Projects/Projects/P3-energy-consumption-mlflow
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**  
   - Use the [UCI Individual household electric power consumption dataset](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption).
   - Place the raw data in the `data/` folder.

4. **Run MLflow UI (optional):**
   ```bash
   mlflow ui
   ```
   Visit [http://localhost:5000](http://localhost:5000) to view experiment tracking.

---

## 🛠️ How to Use

- **Train a model:**  
  Run the training script in `src/` to start an experiment and log results to MLflow.
- **Track experiments:**  
  Use MLflow UI to compare runs, parameters, and metrics.
- **Register models:**  
  Promote the best model to the MLflow Model Registry for version control.

---

## 🛠️ Data Engineering & Experiment Tracking

- **Raw Data:**  
  The dataset is provided in `.txt` format. You’ll need to:
  - Parse and clean the data (handle missing values, convert types, etc.).
  - Engineer features (e.g., rolling averages, time-based features).
  - Split data into train/test sets based on time.

- **Experiment Tracking with MLflow:**  
  - Create multiple data versions by cropping to different time periods or applying different preprocessing pipelines.
  - Log each experiment run in MLflow, including:
    - Data version or preprocessing method used
    - Model parameters and metrics
    - Artifacts (plots, feature importances, etc.)
  - Use MLflow’s model registry to manage and compare models trained on different data versions.

---

## 📚 Skills & Concepts

- Time series regression and feature engineering
- MLflow experiment tracking and model registry
- Modular ML project structure
- Reproducibility and version control

---

## 📄 License

This project is licensed under the MIT License.

---

*Maintained as part of the [ML Engineering Projects](https://github.com/aakashsinghgit/ML-Engineering-Projects) series.*
