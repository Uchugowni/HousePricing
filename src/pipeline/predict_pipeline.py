import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        pass
    def stu_predict(self, features):
        try:
            model_path = 'notebook\model.pkl'
            preprocessor_path = 'notebook\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            logging.info('Exception Occured while stu_predict pipeline ')
            raise CustomException(e, sys)
        
    def GT_predict(self, features):
        try:
            model_path = 'GT_artifacts\model.pkl'
            preprocessor_path = 'GT_artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            logging.info('Exception Occured while GT_predict pipeline ')
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education,
                 lunch : str,
                 test_preparation_course : str,
                 writing_score: int,
                 reading_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.writing_score = writing_score
        self.reading_score = reading_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch" : [self.lunch],
                "test_preparation_course" : [self.test_preparation_course],
                "writing_score" : [self.writing_score],
                "reading_score" : [self.reading_score]
                }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomData(e, sys)
        

class CustomData_for_GT:
    def __init__(self,
                carat : float,
                depth : float,
                table : float,
                x : float,
                y : float,
                z : float,
                cut : str,
                color : str,
                clarity : str):
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_GT_dataframe(self):
        try:
            custm_data_dict ={
                "carat" : [self.carat],
                "depth" : [self.depth],
                "table" : [self.table],
                "x" : [self.x],
                "y" : [self.y],
                "z" : [self.z],
                "cut" : [self.cut],
                "color" : [self.color],
                "clarity" : [self.clarity]
                }
            return pd.DataFrame(custm_data_dict)

        except Exception as e:
            raise CustomException(e, sys)

