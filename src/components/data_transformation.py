import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logging
from src.exception import CustomException

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    num_colums = ["writing_score", "reading_score"]
    cat_colums = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
                ]
    target_column_name = "math_score"
    GT_num_colums = ["carat", "depth", "table", "x", "y", "z"]
    GT_cat_colums = ["cut", "color", "clarity"]
    GT_target_column_name ="price"

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            numerical_columns = self.data_transformation_config.num_colums
            catgorical_columns = self.data_transformation_config.cat_colums
            
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info("Numerical columns are standard scaled completed")
            logging.info(f"Numerical columns are : {numerical_columns}")
            logging.info("Categorical columns are encoded and scaled")
            logging.info(f"Categorical columns are: {catgorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, catgorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            logging.info('Exception Occured while preprocessing object')
            raise CustomException(e, sys)    
    
    def initiate_data_transaformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformation_object()

            input_feature_train_df = train_df.drop(columns=[self.data_transformation_config.target_column_name], axis=1)
            target_feature_train_df=train_df[self.data_transformation_config.target_column_name]

            input_feature_test_df = test_df.drop(columns=[self.data_transformation_config.target_column_name], axis=1)
            target_feature_test_df = test_df[self.data_transformation_config.target_column_name]

            logging.info(f"Applying preprocessing object on traing dataframe and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved Preprocessing object")

            

            return (train_arr, test_arr)
        

        except Exception as e:
            logging.info('Exception Occured while transformation to array')
            raise CustomException(e, sys)





