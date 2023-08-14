from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

filename = 'ParkinsonSVM.pkl'
classifier = pickle.load(open(filename,'rb'))
model = pickle.load(open('ParkinsonSVM.pkl','rb'))

app = Flask(__name__, template_folder= "template") #template folder

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict_value():
    input_features = [int(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    feature_name = ['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)',
                    'MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(db)','Shimmer:APQ3',
                    'Shimmer:APQ5','MDVP:APQ','Shimmer:DDA','NHR','HNR']
    df = pd.DataFrame(features_value, columns = feature_name)
    output = model.predict(df)
    if output == 1:
        resvalue = 'Parkinson is being detected .Please refer a good neurologist for further examination'
    else:
        resvalue = "Parkinson not being detected"
        
    return render_template('parkinsonresult.html', prediction_text='Following Diagnosis is made:{}'. format(resvalue))
    
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080)