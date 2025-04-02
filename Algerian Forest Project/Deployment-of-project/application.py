import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template
# render_template is responsible to finding the URL of the HTML file
application = Flask(__name__)
# we have renamed file to application instead of app as the AWS EBExtension looks for the main file named "application"
app = application

# import ridge regressor and standard scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
scaler_model = pickle.load(open('models/scaler.pkl', 'rb'))


@app.route("/")
def index():
    return render_template('index.html')
# the render_template will try to find the index.html file in the fodler named "templates"

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_data():
    if request.method == 'POST':
        # here every input is taken the name given to thwm in the form in the home.html
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC= float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # To standardize all the inputted values, we send these values to the standard scaler model
        new_data = scaler_model.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_data)

        # here the result is always sent in form of a list and we have only one row, so selected the result[0]
        return render_template('home.html', results = result[0])
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")