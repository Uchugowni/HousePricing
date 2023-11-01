import os 
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass
from src.logger import logging

@dataclass
class transfordatastore:
    pickle_file = os.path.join("artifacts", "proprocess.pkl")

class transform_data:
    def __init__(self):
        self.picklefile_path = transfordatastore()

    def preprocessor_config(self):
        try:
            num_colmns = ["writing_score", "reading_score"]
            cat_colmns = ["gender",
                          "race_ethnicity",
                          "parental_level_of_education",
                          "lunch",
                          "test_preparation_course"]
            logging.info(f"Nummerical {num_colmns} and Categorical {cat_colmns} columns are insert in varable")
            num_pipe = Pipeline(steps=[
                ("Imputer", SimpleImputer(strategy="mean")),
                ("Scaler", StandardScaler())
                ])
            cat_pipe = Pipeline(steps=[
                ("Imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder()),
                ("Scaler", StandardScaler(with_mean=False))
                ])
            logging.info("Nummerical and Categorical features are transformed as per Standard and OneHot")
            preprocesser = ColumnTransformer([("num_pipeline", num_pipe, num_colmns),("cat_pipeline", cat_pipe, cat_colmns)])

            return preprocesser
        except Exception as e:
            raise CustomException(e, sys)
        
    def preprocesser_data(self, train_data, test_data):
        try:
            train_df = pd.read_csv(train_data)
            test_df = pd.read_csv(test_data)
            logging.info("Train and Test splited is imported")
            preprocess_obj = self.preprocessor_config()

            target_column = "math_score"
            train_input_columns = train_df.drop(columns=[target_column], axis=1)
            train_target_column = train_df[target_column]

            test_input_columns = test_df.drop(columns=[target_column], axis=1)
            test_target_column = test_df[target_column]
            logging.info("Train and test data features ara seperated with input and target")
            train_input_array = preprocess_obj.fit_transform(train_input_columns)
            test_input_array = preprocess_obj.transform(test_input_columns)
            logging.info("Target train and test data is transformed as per standard and oneHot")
            save_object(
                file_path = self.picklefile_path.pickle_file,
                obj = preprocess_obj
            )
            logging.info("Transformed format is fitted in pickle file and stored in artifacts folder")
            train_array = np.c_[train_input_array, np.array(train_target_column)]
            test_array = np.c_[test_input_array, np.array(test_target_column)]
            logging.info("After transformed input feature train and test array data is merged and return")
            return (train_array, test_array)
        
        except Exception as e:
            raise CustomException(e, sys)
