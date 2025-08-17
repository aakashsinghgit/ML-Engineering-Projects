# Health Insurance Price Predictor

This project is a **modular ML engineering application** that predicts **health insurance prices** based on user input. It is designed to showcase how to structure ML projects for maintainability, reproducibility, and production-readiness.

It builds on the basics from Project 1, introducing **modular code organization, logging, exception handling, pipelines, and CI/CD practices**. The project also demonstrates how to containerize and deploy ML services.

---

## 🚀 What You’ll Learn

✅ Modular project structure for ML engineering  
✅ Custom logging and exception handling  
✅ Building reusable components and pipelines  
✅ Model training, evaluation, and deployment  
✅ Serving predictions via a Flask web app  
✅ CI/CD with GitHub Actions  
✅ Dockerization for reproducibility

---

## 🗂️ Project Structure

```
P2_HealthInsurance_Price/
├── artifacts/                # Stores intermediate and final artifacts (models, data, etc.)
├── catboost_info/            # CatBoost training logs (if used)
├── end2endML.egg-info/       # Python package info
├── logs/                     # Log files
├── notebook/
│   ├── data/
│   │   └── insurance.csv     # Dataset for EDA and model training
│   └── insurance_pipeline_dev.ipynb  # Notebook for EDA and prototyping
├── readme.md                 # Project overview and instructions
├── requirements.txt          # Python dependencies
├── setup.py                  # Project setup for pip install
├── src/
│   ├── components/           # Modular code for data ingestion, transformation, training, etc.
│   ├── exception.py          # Custom exception handling
│   ├── logger.py             # Logging setup
│   ├── pipelines/            # Training and prediction pipelines
│   ├── utils.py              # Utility functions
│   └── __init__.py
├── templates/                # HTML templates for the Flask app
│   ├── home.html
│   └── index.html
├── app.py                    # Flask web application
└── venv/                     # Virtual environment (not tracked in git)
```

---

## 🧑‍💻 Features

- **Data Ingestion:** Modular class to load and split data.
- **Data Transformation:** Feature engineering and preprocessing.
- **Model Training:** Train regression models to predict insurance prices.
- **Model Evaluation:** Evaluate and compare model performance.
- **Model Deployment:** Save and load models for inference.
- **Web App:** Serve predictions via a Flask web interface.
- **Logging & Exception Handling:** Track pipeline steps and errors.
- **CI/CD:** Automate testing and deployment with GitHub Actions.
- **Dockerization:** Containerize the app for consistent deployment.

---

## 🛠️ Setup & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/aakashsinghgit/ML-Engineering-Projects.git
   cd ML-Engineering-Projects/Projects/P2_HealthInsurance_Price
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model**
   - Run the Jupyter notebook in `notebook/` for EDA and initial training.
   - Use the modular pipeline in `src/pipelines/train_pipeline.py` to train and save the model.

5. **Run the web app**
   ```bash
   python app.py
   ```
   Visit [http://localhost:5000](http://localhost:5000) in your browser.

6. **Run with Docker**
   ```bash
   docker build -t health-insurance-predictor .
   docker run -p 5000:5000 health-insurance-predictor
   ```

7. **Deploy to Production (Free!)**
   - Follow the [DEPLOYMENT.md](DEPLOYMENT.md) guide
   - Deploy to Render with automated CI/CD
   - Your app will be live on the internet!

---

## 📝 Project Steps

1. **Project Setup:**  
   - Initialize git, create `src/`, `setup.py`, and requirements.
   - Install dependencies and set up the package structure.

2. **Modular Code Structure:**  
   - Create components for data ingestion, transformation, and model training.
   - Implement custom exception and logging modules.
   - Build utility functions for saving/loading models.

3. **Model Development:**  
   - Explore data and train models in the notebook.
   - Move logic into modular pipeline scripts.

4. **Web App Integration:**  
   - Build a Flask app to serve predictions.
   - Connect the trained model to the web interface.

5. **CI/CD & Deployment:**  
   - Set up GitHub Actions for automated testing and deployment.
   - Containerize the app with Docker for cloud or local deployment.

---

## 💡 Next Steps

- Try deploying the app to Azure, AWS, or another cloud platform.
- Experiment with different regression models and feature engineering.
- Enhance the web interface for better user experience.

---

## 🏗️ Contribution

Feel free to fork, open issues, or submit pull requests for improvements or new project ideas!

## 📄 License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
*Maintained by [aakashsinghgit](https://github.com/aakashsinghgit)*
