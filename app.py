from flask import Flask, render_template, request, jsonify
from bs4 import BeautifulSoup as bs
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

## import ridge regressor model and scaler file
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))
ridge_model = pickle.load(open('models/RidgeRegressor.pkl', 'rb'))

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        temp = request.form.get('Temperature')
        rh = request.form.get('RH')
        ws = request.form.get('Ws')
        rain = request.form.get('Rain')
        ffmc = request.form.get('FFMC')
        dmc = request.form.get('DMC')
        isi = request.form.get('ISI')
        classes = request.form.get('Classes')
        region = request.form.get('Region')
        test_data = [[temp, rh, ws, rain, ffmc, dmc, isi, classes, region]]
        scaled_test_data = standard_scaler.transform(test_data)
        predicted_data = ridge_model.predict(scaled_test_data)
        print(predicted_data[0][0])
        return render_template('home.html', result = predicted_data[0][0])
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host = '0.0.0.0')
