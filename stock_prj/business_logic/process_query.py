import configparser
import logging
import joblib

from stock_prj.IO.storage_tools import create_bucket, get_model_from_bucket, upload_file_to_bucket


class BusinessLogic:

    def __init__(self, model_creator):
        self._root_bucket = 'model_bucket_hnnu_ycng228'
        self._config = configparser.ConfigParser()
        self._config.read('application.conf')
        self._model_creator = model_creator
        self._create_bucket()

    def get_version(self):
        return self._config['DEFAULT']['version']


    def get_bucket_name(self):
        return f'{self._root_bucket}_{self.get_version().replace(".", "")}'


    def _get_or_create_model(self, ticker, smth):
        log = logging.getLogger()
        model_filename = self.get_model_filename_from_ticker(ticker, smth)
        model = get_model_from_bucket(model_filename, self.get_bucket_name())
                        	           
        if model is None:
            log.warning(f'training model for {ticker}')
            model = self._model_creator.fit(ticker, smth)
            
            with open(model_filename, 'wb') as f:
                joblib.dump(model, f)
            upload_file_to_bucket(model_filename, self.get_bucket_name()) 
        
        #model = self._model_creator.fit(ticker, smth)
        return model


    def get_model_filename_from_ticker(self, ticker, smth):
        return f'{ticker}_'+str(smth)+'.pkl'


    def _create_bucket(self):
        create_bucket(self.get_bucket_name())


    def do_predictions_for(self, ticker, smth):
        
        model = self._get_or_create_model(ticker, smth)        
        predictions = model.predict(ticker, smth)

        return predictions


