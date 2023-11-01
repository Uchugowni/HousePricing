import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.Prac.data_trans import transform_data
from dataclasses import dataclass
from src.Prac.model_train import model_train_data
from src.logger import logging

@dataclass
class store_raw_data():
    raw_data = os.path.join("artifacts", "raw_data.csv")
    train_data = os.path.join("artifacts", "train_data.csv")
    test_data = os.path.join("artifacts", "test_data.csv")

class DataIngestion():
    def __init__(self):
        self.data_path = store_raw_data()

    def segreegate_raw_data(self):
        try:
            df = pd.read_csv("notebook\stud.csv")
            logging.info("Dataframe is readed")
            os.makedirs(os.path.dirname(self.data_path.raw_data), exist_ok=True)
            df.to_csv(self.data_path.raw_data, index=False, header=True)

            train_set, test_set = train_test_split(df, test_size=0.33, random_state=100)
            logging.info("Data split in to train and test with raw")
            train_set.to_csv(self.data_path.train_data, index=False, header=True)
            test_set.to_csv(self.data_path.test_data, index=False, header=True)
            logging.info("Raw train and test data splited and stored in artifacts folder in local")
            return(
                self.data_path.train_data, self.data_path.test_data
            )
        
        except Exception as e:
            raise CustomException(e, sys)

        
if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.segreegate_raw_data()
    transf_data = transform_data()
    train_array, test_array = transf_data.preprocesser_data(train_data, test_data)
    print("Code success")
    mt_obj = model_train_data()
    mt_obj.model_training(train_array, test_array)
    

