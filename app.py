import numpy as np
from flask import Flask, request, jsonify, render_template   #flask ; host my model , render_template : redirect to home page for inout to get ouput
import pickle

app = Flask(__name__)  #initailization
#app.run("localhost", "9999", debug=True)
model = pickle.load(open('model.pkl', 'rb'))  #read mode model

@app.route('/')
def home():
    return render_template('index.html')      #indirect to home file

@app.route('/predict',methods=['POST'])      #look at from tag of index page , click button to run this api code
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]   #read values from index page
    final_features = [np.array(int_features)]               #convert into array
    prediction = model.predict(final_features)    #prediction

    #output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Customer 1 for Exited  $ {}'.format(prediction)) #prediction_text is tag og index page
    
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request, this is an json values pass form request.py file , and get it output
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True , port=9999)