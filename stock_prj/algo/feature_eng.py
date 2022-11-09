from csaps import csaps
import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import shift

def create_features(df_stock, smth=1.0, lss=True, nlags=4):
    df_resampled = df_stock.copy()
          
    y1 = df_resampled['close']
                  
    # smoothing the curve
    y2 = df_resampled['close'].shift(1)
    
    y2[0]=y2[1]
    LNG = len(y1) 
    
    x1 = range(1,LNG+1)  
    xi = np.linspace(x1[0], x1[-1], LNG)
    sp = csaps(x1, y2, xi, smooth=smth)    
    
    _df = pd.DataFrame({'close':y1, 'close_spl':sp})
    _df['lag_0'] = _df['close'].shift(0)
    _df['lag_1'] = _df['close'].shift(1)
    # Y-value
    _df['buy_sell'] = np.where(_df['lag_0'] > _df['lag_1'], 1, 0)     

    slag_col_names = []
    for i in range(1, nlags + 2):
        _df['slag_' + str(i)] = _df['close_spl'].shift(i)
        slag_col_names.append('slag_' + str(i))    
    
    df = _df[slag_col_names]
   
    ders_col_names = []
    for i in range(1, nlags + 1):
        _df['der_' + str(i)] = _df['slag_' + str(i)]-_df['slag_' + str(i+1)]
        ders_col_names.append('der_' + str(i))    
    
    df = _df[ders_col_names]
    df['buy_sell'] = _df['buy_sell']
    
    df['chg_der'] = 0
    df['chg_der'] = np.where(df['der_2'] < 0, np.where(df['der_1'] > 0, 1, 0), 0)     
    df['inc_der'] = 0
    df['inc_der'] = np.where(df['der_2'] > 0, np.where(df['der_1'] > df['der_2'], 1, 0), 0)     
       
    df = df.dropna(axis=0)    
    print(df.head())   

    return df


def create_X_Y(df_lags):
    Y = df_lags[['buy_sell']]
    X = df_lags.drop(['buy_sell'], axis=1)    
    return X, Y

