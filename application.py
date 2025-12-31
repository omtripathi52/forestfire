## Project is running in virtual environment (A private Python + packages folder just for one project) for this project to run: py -3.12 -m venv venv
## To run this, Change directory : cd "D:\Coding Folder\Python\9. ML Code (Original)\1. Linear Regression\2. Linear Regression Project" then python application.py


import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import pickle

## Add /predictdata after the link

application = Flask(__name__)
app=application

## import ridge regresor model and standard scaler pickle

## ridge_model=pickle.load(open('models/ridge.pkl','rb'))
## standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

# Absolute base directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to model files
RIDGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "ridge.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Load models
ridge_model = pickle.load(open(RIDGE_MODEL_PATH, "rb"))
standard_scaler = pickle.load(open(SCALER_PATH, "rb"))

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")
