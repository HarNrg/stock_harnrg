from datetime import datetime, timedelta
from yahoo_fin import stock_info as si

'''
def get_last_stock_price(ticker, last=False):
    if last:
        now = datetime.now()
        start_date = now - timedelta(days=30)
        return si.get_data(ticker, start_date)
    
    return si.get_data(ticker)
'''   
   
def get_last_stock_price(ticker, last=False):
    
    if last:
        start_dt = "6/01/2022"       
        end_dt = "9/01/2022" #datetime.now() 
        #now = datetime.now()
        #start_date = now - timedelta(days=30)
        return si.get_data(ticker, start_date = start_dt, end_date = end_dt)
    else:
        start_dt = "1/01/2020"       
        end_dt   = "6/01/2022"       
        return si.get_data(ticker, start_date = start_dt, end_date = end_dt)
        
