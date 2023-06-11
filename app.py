from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
model=pickle.load(open('Model\House_Price.pickle', 'rb'))
preprocessor=pickle.load(open('Model\scaler_obj.pickle','rb'))

@app.route('/predictHousePrice',methods=['GET','POST'])
# Features
# 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT' 
def predict():
    price_of_house=None
    if request.method=='GET':
        return render_template('home.html')
    else:
      #print(request.items())
      data={
        'CRIM' : float(request.form.get('CRIM')), 
        'ZN' : float(request.form.get('ZN')),
        'INDUS' : float(request.form.get('INDUS')),
        'CHAS' : float(request.form.get('CHAS')),
        'NOX' : float(request.form.get('NOX')),
        'RM' : float(request.form.get('RM')),
        'AGE' : float(request.form.get('AGE')),
        'DIS' : float(request.form.get('DIS')),
        'RAD' : float(request.form.get('RAD')),
        'TAX' : float(request.form.get('TAX')),
        'PTRATIO' : float(request.form.get('PTRATIO')),
        'B' : float(request.form.get('B')),
        'LSTAT' : float(request.form.get('LSTAT'))
       }
      print(data)
      input_features=pd.DataFrame(data,index=[0])
      input_values=np.reshape(np.array(input_features.loc[0]),[1,-1])
      scaled_input_values=preprocessor.transform(input_values)
      price_of_house=model.predict(scaled_input_values)
      return render_template('home.html',price=price_of_house)

if __name__ == '__main__':
    app.run(debug=True)