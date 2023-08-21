
import inflection
import pandas as pd
import math
import datetime
import pickle
import numpy as np


class Rossmann(object):
    
    def __init__(self):
        
        self.competition_distance_scaler = pickle.load(open('parameter/competition_distance_scaler.pkl', 'rb'))
        self.competition_time_month_scaler = pickle.load(open('parameter/competition_time month_scaler.pkl', 'rb'))
        self.promo_time_month_scaler = pickle.load(open('parameter/promo_time_week_scaler.pkl', 'rb'))
        self.year_scaler = pickle.load(open('parameter/year_scaler.pkl', 'rb'))
        self.store_type_encoder = pickle.load(open('parameter/store_type_encoder.pkl', 'rb'))
        
        
        
    def data_cleaning(self, df):
        
        ## 1.1 RENAME COLUMNS
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo',
                    'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
                    'CompetitionDistance', 'CompetitionOpenSinceMonth',
                    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                    'Promo2SinceYear', 'PromoInterval']

        snakecase = lambda x: inflection.underscore(x)
        cols_new = list(map(snakecase, cols_old))

        #rename
        
        
        df.columns = cols_new

        ## 1.3 Data Types
        df['date'] = pd.to_datetime( df['date'] )

        ## 1.5 Fillout NA
        
        #competition_distance
        df['competition_distance'] = df['competition_distance'].apply(lambda x: 200000.0 if math.isnan(x) else x)

        #competition_open_since_month   

        df['competition_open_since_month'] = df[['competition_open_since_month' , 'date']].apply( lambda x: x['date'].month 
                                                if math.isnan( x['competition_open_since_month'] ) 
                                                else x['competition_open_since_month'],
                                                axis=1 )

        #competition_open_since_year     
        df['competition_open_since_year'] = df[['competition_open_since_year' , 'date']].apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else x['competition_open_since_year'], axis = 1)                       

        #promo2_since_week              
        df['promo2_since_week'] = df[['promo2_since_week' , 'date']].apply( lambda x: x['date'].week if math.isnan( x['promo2_since_week'] ) else x['promo2_since_week'], axis=1 )

        #promo2_since_year               
        df['promo2_since_year'] = df[['promo2_since_year' , 'date']].apply( lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else x ['promo2_since_year'], axis= 1)

        #promo_interval 
        month_map = {1: 'Jan',  2: 'Fev',  3: 'Mar',  4: 'Apr',  5: 'May',  6: 'Jun',  7: 'Jul',  8: 'Aug',  9: 'Sept',  10: 'Oct', 11: 'Nov', 12: 'Dec'}

        df['promo_interval'].fillna(0, inplace= True)
        df['month_map'] = df['date'].dt.month.map(month_map)
        df['is_promo'] = df[['promo_interval', 'month_map']].apply(lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis =1) 
        ## 1.6 Change Data Types
        # competiton
        df['competition_open_since_month'] = df['competition_open_since_month'].astype( int )
        df['competition_open_since_year'] = df['competition_open_since_year'].astype( int )
            
        # promo2
        df['promo2_since_week'] = df['promo2_since_week'].astype( int )
        df['promo2_since_year'] = df['promo2_since_year'].astype( int )
        
        return df
        
    def feature_engineering(self, df):
        
        ### 2.4. Feature Engineering
        # year
        df['year'] = df['date'].dt.year

        # month
        df['month'] = df['date'].dt.month

        # day
        df['day'] = df['date'].dt.day

        # week of year
        df['week_of_year'] = df['date'].dt.isocalendar().week

        # year week
        df['year_week'] = df['date'].dt.strftime( '%Y-%W' )

        # competition since
        df['competition_since'] = df.apply( lambda x: datetime.datetime( year=x['competition_open_since_year'], month=x['competition_open_since_month'],day=1 ), axis=1 )
        df['competition_time_month'] = ( ( df['date'] - df['competition_since'] )/30 ).apply( lambda x: x.days ).astype( int )

        # promo since
        df['promo_since'] = df['promo2_since_year'].astype( str ) + '-' + df['promo2_since_week'].astype( str )
        df['promo_since'] = df['promo_since'].apply( lambda x: datetime.datetime.strptime( x + '-1', '%Y-%W-%w' ) - datetime.timedelta( days=7 ) )
        df['promo_time_week'] = ( ( df['date'] - df['promo_since'] )/7 ).apply( lambda x: x.days ).astype( int )

        # assortment
        df['assortment'] = df['assortment'].apply( lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended' )

        # state holiday
        df['state_holiday'] = df['state_holiday'].apply( lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day' )

        ## 3.0. PASSO 03 - FILTRAGEM DE VARIÁVEIS
        ### 3.1. Filtragem das Linhas
        # Quero apenas dados onde a loja esta aberta e que valor de vendas seja maior que 0

        ### 3.2. Selecao das Colunas
        
        # Excluir colunas na qual eu não terei durante o processo de previsão
        cols_drop = ['open', 'promo_interval', 'month_map']
        df = df.drop( cols_drop, axis=1 )
        return df
    
    def data_preparation(self, df):
        
        ## 5.1 Rescaling

        # competition distance
        df['competition_distance'] = self.competition_distance_scaler.fit_transform(df[['competition_distance']].values)

        # competition time month
        df['competition_time_month'] = self.competition_time_month_scaler.fit_transform( df[['competition_time_month']].values )

        # promo time week
        df['promo_time_week'] = self.promo_time_month_scaler.fit_transform( df[['promo_time_week']].values )

        # year
        df['year'] = self.year_scaler.fit_transform( df[['year']].values )

        ## 5.2 Transformacao
        ### 5.2.1 Encoding
        
        # state_holiday - One Hot Encoding
        df = pd.get_dummies( df, prefix=['state_holiday'], columns=['state_holiday'] )

        # store_type - Label Encoding
        
        df['store_type'] = self.store_type_encoder.fit_transform( df['store_type'] )

        # assortment - Ordinal Encoding
        assortment_dict = {'basic': 1,  'extra': 2, 'extended': 3}
        df['assortment'] = df['assortment'].map( assortment_dict )
        
        ### 5.2.2 Response Variable Transformation
        
        ### 5.2.3 Nature Transformation
        # day of week
        df['day_of_week_sin'] = df['day_of_week'].apply( lambda x: np.sin( x * ( 2. * np.pi/7 ) ) )
        df['day_of_week_cos'] = df['day_of_week'].apply( lambda x: np.cos( x * ( 2. * np.pi/7 ) ) )

        # month
        df['month_sin'] = df['month'].apply( lambda x: np.sin( x * ( 2. * np.pi/12 ) ) )
        df['month_cos'] = df['month'].apply( lambda x: np.cos( x * ( 2. * np.pi/12 ) ) )

        # day 
        df['day_sin'] = df['day'].apply( lambda x: np.sin( x * ( 2. * np.pi/30 ) ) )
        df['day_cos'] = df['day'].apply( lambda x: np.cos( x * ( 2. * np.pi/30 ) ) )

        # week of year
        df['week_of_year_sin'] = df['week_of_year'].apply( lambda x: np.sin( x * ( 2. * np.pi/52 ) ) )
        df['week_of_year_cos'] = df['week_of_year'].apply( lambda x: np.cos( x * ( 2. * np.pi/52 ) ) )
        
        cols_select = ['store',
                        'promo',
                        'store_type',
                        'assortment',
                        'competition_distance',
                        'competition_open_since_month',
                        'competition_open_since_year',
                        'promo2',
                        'promo2_since_week',
                        'promo2_since_year',
                        'day_of_week_cos',
                        'day_of_week_sin',
                        'month_cos',
                        'day_sin',
                        'day_cos',
                        'month_sin']
        return df[cols_select]
    
    def get_prediction(self, model, original_data, test_data):
        
        #prediction
        pred = model.predict(test_data)
        
        # join pred into the original data
        
        original_data['prediction'] = np.expm1(pred)
        
        return original_data.to_json(orient= 'records', date_format= 'iso')