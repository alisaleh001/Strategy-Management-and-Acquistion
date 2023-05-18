from flask import Flask , request , jsonify , render_template
import pickle
import numpy as np
from sklearn.metrics import accuracy_score

app = Flask(__name__,template_folder='Templates')

# Load pickle model
model = pickle.load(open(r"Ali's Project\model.pkl" , "rb"))

@app.route('/')
def home():
    return render_template("titanic.html")



@app.route('/predict',methods=['POST']) 
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if output == 1:
        output="Survived"
    else:
        output="not Survived I'm sorry for that "

    

    return render_template('titanic.html', prediction_text='The person who want to check servive prediction will '+ output)

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
