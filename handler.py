# imports

from flask import request, Flask, Response
import pickle
import json
import pandas as pd
from rossmann.Rossmann import Rossmann


# loading model

model = pickle.load(open('C:/Users/55329/Documents/3.Repos/Rossmann-sales/model/model_xgb_tuned.pkl', 'rb'))

# initialize API

app = Flask(__name__)

@app.route('/rossmann/predict', methods = ['POST'])
 
def rossmann_predict():
    """
    Faz uma previsão de vendas utilizando o modelo treinado da classe Rossmann.

    Essa função recebe um conjunto de dados de entrada no formato JSON, realiza a limpeza, engenharia de
    características e preparação dos dados usando a classe Rossmann. Em seguida, utiliza o modelo treinado
    para fazer a previsão de vendas.

    Retorna:
        pandas.DataFrame: Um DataFrame contendo as previsões de vendas para cada loja.

    Exemplo:
        >>> test_json = '{"Store": 1, "DayOfWeek": 3, "Open": 1, "Promo": 0, "StateHoliday": "0", ...}'
        >>> result_df = rossmann_predict()
    """
    test_json = request.get_json()
    test_json = json.loads(test_json)
    if test_json:
        if isinstance(test_json, dict):  # Uma linha única
            test_raw = pd.DataFrame(test_json, index=[0])
        else:  # Múltiplas linhas
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        # Instaciar classe Rossmann
        pipeline = Rossmann()

        # data cleaning
        df1 = pipeline.data_cleaning(test_raw)

        # feature engineering
        df2 = pipeline.feature_engineering(df1)

        # data preparation
        df3 = pipeline.data_preparation(df2)

        # prediction
        df_response = pipeline.get_prediction(model=model, test_data=df3, original_data=test_raw)

        return df_response
    else:
        return Response('[]', status=200, mimetype='application/json')

    
if __name__ == '__main__':

    app.run( '0.0.0.0' , debug=True)