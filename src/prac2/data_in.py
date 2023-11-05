import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.prac2.data_tr import DataTrans_process
from src.prac2.model_tr import ModelTrain_Process


class DataIn_Path:
    raw_data = os.path.join('artifacts', 'raw_p.csv')
    train_data = os.path.join('artifacts', 'train_p.csv')
    test_data = os.path.join('artifacts', 'test_p.data')

class DataIn_Process:
    def __init__(self):
        self.dir_name_path = DataIn_Path()
    def split_train_test(self):
        try:
            df = pd.read_csv('notebook\stud.csv')
            os.makedirs(os.path.dirname(self.dir_name_path.raw_data), exist_ok=True)
            df.to_csv(self.dir_name_path.raw_data, index=False, header=True)
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=40)
            train_set.to_csv(self.dir_name_path.train_data, index=False, header=True)
            test_set.to_csv(self.dir_name_path.test_data, index=False, header=True)
            print("Train and test both set's are created")
            return (
                self.dir_name_path.train_data, self.dir_name_path.test_data
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    obj = DataIn_Process()
    train_data, test_data = obj.split_train_test()
    
    dt_obj = DataTrans_process()
    train_arr, test_arr = dt_obj.preprocess_train_test(train_data, test_data)

    mt_obj = ModelTrain_Process()
    print(mt_obj.predict_train_test(train_arr, test_arr))
