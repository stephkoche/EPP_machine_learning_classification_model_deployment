import numpy as np
from flask import Flask, request, jsonify, render_template
from jinja2 import escape
import xgboost as xgb
#import pickle

app = Flask(__name__)
#model = pickle.load(open('models/promotion-model.pkl', 'rb'))
model = xgb.XGBClassifier()
# model = xgb.Booster()
model.load_model("models/promotion-model.bin")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='The Employee class is {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)