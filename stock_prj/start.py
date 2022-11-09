import logging

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from stock_prj.business_logic.process_query import BusinessLogic
from stock_prj.IO.get_data_from_yahoo import get_last_stock_price
from stock_prj.algo.new_model import Stock_model
from stock_prj.IO.toFile import summary_file
    
def create_business_logic(ticker):    
    
    data_fetcher = get_last_stock_price   # ref to get_last_stock_price
    log = logging.getLogger()
    model = LogisticRegression(tol=0.0001, max_iter=200)
            
    modeProd = True
    prediction = ""       
    
    if(modeProd):
      #Mode production
      smth = 0.6
      #file_open = 'summary.csv'
      #with open(file_open, 'a+') as file:
      #  file.write(str(smth) + ';') 
                           
      bl = BusinessLogic(Stock_model(data_fetcher, log, model))        
      prediction = bl.do_predictions_for(ticker, smth)
    
    else:
      #mode testing smooth coef.     
      arr_smth = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
                                    
      for smth in arr_smth: 
        #file_open = 'summary.csv'
        #with open(file_open, 'a+') as file:
        #  file.write(str(smth) + ';')                 
        bl = BusinessLogic(Stock_model(data_fetcher, log, model))        
        prediction = bl.do_predictions_for(ticker, smth)
                      
    return prediction  



