from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from src.pipelines.predict_pipeline import PredictPipeline, CustomData


app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')    

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        # Get the input values from the form
        return render_template('home.html')
    else:
        data = CustomData(
            age=int(request.form.get('age', 0)),
            sex=str(request.form.get('sex', '')),
            bmi=float(request.form.get('bmi', 0.0)),
            children=int(request.form.get('children', 0)),
            smoker=str(request.form.get('smoker', '')),
            region=str(request.form.get('region', ''))
        )
        # Convert the input data to a DataFrame
        data_df = data.get_data_as_dataframe()
        # Predict using the PredictPipeline
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(data_df)
        # Format the result: Rs <amount> per year, rounded to 2 decimals
        formatted_result = f"Rs {results[0]:.2f} per year"
        return render_template('home.html', results=formatted_result)
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)