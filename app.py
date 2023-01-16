import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)  # starting a Flask app
model = pickle.load(open('model.pkl', 'rb'))  # loading a pickle file


@app.route('/')
def home():
    # return 'Hello World'
    return render_template('home.html')


@app.route('/predict_api', methods=[
    'POST'])  # create an api whose name is 'predict_api'(or 'predict_apibatch' for batch input, and then input dataframe
def predict_api():  # we will not create any html page, we will post requests from postman
    data = request.json['data']  # 'data' because in postman the format that i send is 'data': value
    print(
        data)  # the entire key value pair (1 record i.e. 1 row) of inputdata that i will put in postman will be printed
    new_data = [list(
        data.values())]  # converting data(which is in 1D) into 2D data so that i can use my generalized created model
    output = model.predict(new_data)[0]
    return jsonify(output)  # convert into json


@app.route('/predict', methods=['POST'])
def predict():  # for html page (almost same as above predict_api code)
    data = [float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)

    output = model.predict(final_features)[0]
    print(output)
    return render_template('home.html', prediction_text='Airfoil pressure is {}'.format(output))


if __name__ == "__main__":  # this is where point of execution starts
    app.run(debug=True)