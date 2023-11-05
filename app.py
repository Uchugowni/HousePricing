import pickle
import json
from joblib import dump, load
#from django.shortcuts import render
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
#from waitress import serve
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


app = Flask(__name__)
lrmodel = load('lrmodel.joblib')
scaler = load('scaling.joblib')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = lrmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data =[float(x) for x in request.form.values()]
    sc_data = scaler.transform(np.array(data).reshape(1, -1))
    print(sc_data)
    lr_output= lrmodel.predict(sc_data)[0]
    return render_template("home.html", prediction_text="House Price is {}".format(lr_output))
    
@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=request.form.get('reading_score'),
            writing_score=request.form.get('writing_score')
        )
        pred_df = data.get_data_as_data_frame()
        predpipeline = PredictPipeline()
        result = predpipeline.predict(pred_df)
        return render_template('index.html', results = result)
    

if __name__=="__main__":
    app.run(debug=True)