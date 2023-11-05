import os
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.utils import save_object


class DataTrans_path:
    preprocess_path  = os.path.join('artifacts', 'preprocess_p.pkl')

class DataTrans_process:
    def __init__(self):
        self.dirname_path = DataTrans_path()

    def data_preprocessing(self):
        try:
            num_pipeline = Pipeline(steps=[
                ("Imputer", SimpleImputer(strategy="mean")),
                ("Scaler", StandardScaler())])
            cat_pipeline = Pipeline(steps=[
                ("Imputer", SimpleImputer(strategy="most_frequent")),
                ("Encoder", OneHotEncoder()),
                ("Scaler", StandardScaler(with_mean=False))])

            num_columns = ("reading_score", "writing_score")
            cat_columns = ("gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course")

            preprocessor = ColumnTransformer([("Num_Columns", num_pipeline, num_columns),("Cat_Columns", cat_pipeline, cat_columns)])
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)



    def preprocess_train_test(self, train_data, test_data):
        try:
            train_df = pd.read_csv(train_data)
            test_df = pd.read_csv(test_data)

            input_features_train_df = train_df.drop("math_score", axis=1)
            target_feature_train_df = train_df["math_score"]
            input_features_test_df = test_df.drop("math_score", axis=1)
            target_feature_test_df = test_df["math_score"]

            preprocessor_obj = self.data_preprocessing()

            input_train_array = preprocessor_obj.fit_transform(input_features_train_df)
            input_test_array = preprocessor_obj.transform(input_features_test_df)

            save_object(
                file_path=self.dirname_path.preprocess_path,
                obj=preprocessor_obj
                )
            
            train_array = np.c_[input_train_array, np.array(target_feature_train_df)]
            test_array = np.c_[input_test_array, np.array(target_feature_test_df)]
            print("Train and test array both are retuned")

            return (train_array, test_array)
        except Exception as e:
            raise CustomException(e, sys)

        
