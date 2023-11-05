import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.utils import load_pfile, load_object


class prediction_pipeline:
    def __init__(self):
        pass
    def newdata_prediction(self, newdataframe):
        try:
            preprocess_pkl = 'artifacts\preprocess_p.pkl'
            model_pkl = 'artifacts\model_p.pkl'
            preprocess = load_object(file_path=preprocess_pkl)
            model = load_object(file_path=model_pkl)
            scaled = preprocess.transform(newdataframe)
            pred_newdf = model.predict(scaled)
            return pred_newdf
        except Exception as e:
            raise CustomException (e, sys)

        


class Custom_Data:
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
            data_input = {
                "gender" : [self.gender],
                "race_ethnicity" : [self.race_ethnicity],
                "parental_level_of_education" : [self.parental_level_of_education],
                "lunch" : [self.lunch],
                "test_preparation_course" : [self.test_preparation_course],
                "writing_score" : [self.writing_score],
                "reading_score" : [self.reading_score]
                
            }
            return pd.DataFrame(data_input)
        except Exception as e:
            raise CustomException (e, sys)