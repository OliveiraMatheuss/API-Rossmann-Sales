# imports

from flask import request, Flask, Response
import pickle
import os
import pandas as pd
from rossmann.Rossmann import Rossmann
import json


# loading model

model = pickle.load(open('model/model_xgb_tuned.pkl', 'rb'))

# initialize API

app = Flask(__name__)

@app.route('/rossmann/predict', methods = ['GET','POST'])

def rossmann_predict():
    test_json = request.get_json()
    test_json = json.loads(test_json)
    
    if test_json:
        if isinstance(test_json, dict): # Uma linha unica 
            test_raw = pd.DataFrame( test_json , index = [0])
            
        else: # Multiplas linhas
            test_raw = pd.DataFrame( test_json, columns= test_json[0].keys())
            
        # Instaciar classe Rosmann
        pipeline = Rossmann()
        
        # data cleaning
        df1 = pipeline.data_cleaning(test_raw)
        
        # feature engineering 
        df2 = pipeline.feature_engineering(df1)
        
        # data prepation
        df3 = pipeline.data_preparation(df2)
        
        # prediction
        df_response = pipeline.get_prediction(model= model, test_data = df3, original_data=test_raw)
        
        return df_response
    else:
        return Response( '{}', status=200, mimetype='application/json' )
    
if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run( host = '0.0.0.0', port = port, debug=True )