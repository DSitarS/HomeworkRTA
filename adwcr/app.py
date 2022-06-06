from flask import Flask
from flask_restful import Resource, Api
from flask import request, jsonify
import joblib
import numpy as np
from train import Perceptron
app = Flask(__name__)


@app.route('/')
def home():
    return "Aplikacja ze srodowiskiem produkcyjnym API"



@app.route('/api/predict', methods=['GET'])

def predict():
    sepal_length = float(request.args.get('sl'))
    sepal_width = float(request.args.get('sw'))
    petal_length = float(request.args.get('pl'))
    petal_width = float(request.args.get('pw'))


    features = [sepal_length,
                sepal_width,
                petal_length,
                petal_width]


    model = joblib.load('model.pkl')

    predicted_class = int(model.predict(np.array(features, dtype=np.float32)))
    return jsonify(features=features, predicted_class=predicted_class)




if __name__ == '__main__':
    app.run(port='3333',host='0.0.0.0')