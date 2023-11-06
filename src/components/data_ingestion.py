import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:

    GT_main_path = 'GT_artifacts\GTtrain.csv'
    main_path = 'notebook\stud.csv'

    #if any drop column mention or else remove name in drop column
    drop_column = "id"

    train_data_path = os.path.join('artifacts', "train.csv")
    test_data_path = os.path.join('artifacts', "test.csv")
    raw_data_path = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method or component")
        try:
            df = pd.read_csv(self.ingestion_config.main_path)
            logging.info("Read dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            #ndf = df.drop(columns=self.ingestion_config.drop_column, axis=1)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info('Exception Occured while data ingestion method')
            raise CustomException(e, sys)

if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transaformation = DataTransformation()
    train_arr,test_arr= data_transaformation.initiate_data_transaformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
