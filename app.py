from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictionPipeline

application=Flask(__name__)

app=application

# Route for first page
@app.route("/")
def index():
    return render_template('index.html')

# Route for prediction page
@app.route("/predict", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

        df_pred = data.get_data_as_data_frame()
        print(df_pred)
        print(f'Before prediction')

        predict_pipeline = PredictionPipeline()
        results = predict_pipeline.prediction(df_pred)
        print(f'After prediction')

        return render_template('home.html', results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)