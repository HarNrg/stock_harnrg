#import logging

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from stock_prj.algo.feature_eng import create_features
from stock_prj.algo.feature_eng import create_X_Y

from csaps import csaps 
import numpy as np
import pandas as pd


class Stock_model(BaseEstimator, TransformerMixin):

    def __init__(self, data_fetcher, log, model):
        self._log = log     
        self._model = model
        self._data_fetcher = data_fetcher
        
        
    def fit(self, X, smth=1.0):         
        data = self._data_fetcher(X, last=False)              
        
        df_features = create_features(data, smth, True)
        df_features, Y = create_X_Y(df_features)     
        Y = np.ravel(Y, order='C')         
        
        df_features_train, df_features_val, Y_train, Y_val = train_test_split(df_features, Y, test_size=0.6, random_state=None, shuffle=False, stratify=None)
        
        self._model.fit(df_features_train, Y_train)
                
        predictions = self._model.predict(df_features_val)
        bal_acc = balanced_accuracy_score(Y_val, predictions)       
            
        #file_open = 'summary.csv'
        #with open(file_open, 'a+') as file:
        #    file.write(str(np.round(np.mean(bal_acc),4))+';') 
                     
        return self


    def predict(self, X, smth=1.0):
        data = self._data_fetcher(X, last=True)     
        df_features = create_features(data, smth, False)

        df_features, Y = create_X_Y(df_features) 
        Y = np.ravel(Y, order='C')
        
        predictions = self._model.predict(df_features)        
        bal_acc = balanced_accuracy_score(Y, predictions)
        
        #file_open = 'summary.csv'
        #with open(file_open, 'a+') as file:
        #    file.write(str(np.round(np.mean(bal_acc),4))+'\n')             
       
        buy_sell = 'buy' if predictions.flatten()[-1]==1 else 'sell'

        return buy_sell
