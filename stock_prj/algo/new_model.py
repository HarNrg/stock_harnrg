import logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from csaps import csaps
import numpy as np
import pandas as pd

def create_features(df_stock, nlags=10):
    df_resampled = df_stock.copy()
    #lags_col_names = []
    #for i in range(nlags + 1):
    #    df_resampled['lags_' + str(i)] = df_resampled['close'].shift(i)
    #    lags_col_names.append('lags_' + str(i))
    #df = df_resampled[lags_col_names]
          
    y1 = df_resampled['close']
    LNG = len(y1)
    x1 = range(1,LNG+1)    
    xi = np.linspace(x1[0], x1[-1], LNG)
    sp = csaps(x1, y1, xi, smooth=0.8)    
    
    _df = pd.DataFrame({'close':y1, 'close_spl':sp}) 
    _df['lag_1'] = _df['close'].shift(1)
             
    _df['slag_0'] = _df['close_spl'].shift(0)
    _df['slag_1'] = _df['close_spl'].shift(1)
    _df['slag_2'] = _df['close_spl'].shift(2)  
    _df['slag_3'] = _df['close_spl'].shift(3)           
    _df['der_0'] = _df['slag_0']-_df['slag_1']
    _df['der_1'] = _df['slag_1']-_df['slag_2']
    _df['buy_sell'] = np.where(_df['close'] > _df['lag_1'], 1, 0) 
    
    _df = _df.dropna(axis=0)
    _df = _df.drop(['close', 'close_spl', 'lag_1', 'slag_0', 'slag_1', 'slag_2', 'slag_3'], axis=1)
    print(_df.head(5))    

    return _df


def create_X_Y(df_lags):
    #X = df_lags.drop('lags_0', axis=1)
    #Y = df_lags[['lags_0']]
    Y = df_lags[['buy_sell']]
    X = df_lags.drop(['buy_sell'], axis=1)    
    return X, Y


class Stock_model(BaseEstimator, TransformerMixin):

    def __init__(self, data_fetcher):
        self.log = logging.getLogger()
        self.lr = LogisticRegression() #LinearRegression()
        self._data_fetcher = data_fetcher
        self.log.warning('here')

    def fit(self, X, Y=None):         
        data = self._data_fetcher(X, last=False)        
        print("Data fix size: ", len(data))
        df_features = create_features(data,4)
        df_features, Y = create_X_Y(df_features)
        Y = np.ravel(Y, order='C')
         
        self.lr.fit(df_features, Y)
        return self

    def predict(self, X, Y=None):
        data = self._data_fetcher(X, last=True)     
        print("Data predict size: ", len(data))
        #print(data)
        df_features = create_features(data,4)
        #print(df_features)
        df_features, Y = create_X_Y(df_features) 
        Y = np.ravel(Y, order='C')               
        predictions = self.lr.predict(df_features)
        
        bal_acc = balanced_accuracy_score(Y, predictions)
        print('balance_accuracy: ' + str(np.round(bal_acc,4)))
        
        buy_sell = 'buy' if predictions.flatten()[-1]==1 else 'sell'

        return buy_sell
