from flask import Flask,render_template,request
import numpy as np
import joblib
app=Flask(__name__)
model=joblib.load("str.h5")
@app.route('/')

def upload():
    return render_template('upload.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features=np.array(int_features).reshape(1,10)
    print(final_features)
    prediction=model.predict(final_features)
    L_collection={0:"No stroke",1:"stroke"}
    result=L_collection[prediction[0]]
    print(result)
    return render_template('result.html',prediction_text=f"YOU HAVE:{result}")

if __name__=="__main__":
    app.run(debug=True)