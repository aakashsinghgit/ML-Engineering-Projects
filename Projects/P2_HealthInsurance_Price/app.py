from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd

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
            age=int(request.form.get('age')),
            sex= request.form.get('sex'),
            bmi=float(request.form.get('bmi')),
            children=int(request.form.get('children')),
            smoker=request.form.get('smoker'),
            region=request.form.get('region')
        )
        # Convert the input data to a DataFrame
        data_df = data.get_data_as_dataframe()
        # Predict using the PredictPipeline
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(data_df)
        # Render the results in the template
        return render_template('home.html', results=results[0])
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)