import os
import sys
import pickle
import numpy as np
import pandas as pd 
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
    
    
def evaluate_models(x_train, y_train, x_test, y_test, models, param):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(estimator=model, param_grid=para, cv=3)
            gs.fit(x_train, y_train)
            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)

            Y_train_pred = model.predict(x_train)
            Y_test_pred = model.predict(x_test)

            train_model_r2_score = r2_score(y_train, Y_train_pred)
            test_model_r2_score = r2_score(y_test, Y_test_pred)
            logging.info(f"modela name --> {model} trann-r2-score {train_model_r2_score} and test_model_r2_score {test_model_r2_score}")

            report[list(models.keys())[i]]=test_model_r2_score
        return report
    
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(x_train, y_train, x_test, y_test, models, param):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            para = param[list(models.keys())[i]]
            gscv = GridSearchCV(estimator=model, param_grid=para, cv=3)
            gscv.fit(x_train, y_train)
            model.set_params(**gscv.best_params_)


            model.fit(x_train, y_train)

            Y_train_pred = model.predict(x_train)
            Y_test_pred = model.predict(x_test)

            train_model_r2_score = r2_score(y_train, Y_train_pred)
            test_model_r2_score = r2_score(y_test, Y_test_pred)
            logging.info(f"modela name --> {model} trann-r2-score {train_model_r2_score} and test_model_r2_score {test_model_r2_score}")

            report[list(models.keys())[i]]=test_model_r2_score
        return report
    
    except Exception as e:
        raise CustomException(e, sys)

def dump_dfile(filepath, obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)

        with open(filepath, "wb") as file_obj:
            dill.dump(obj, filepath)

    except Exception as e:
        raise CustomException (e, sys) 

def load_pfile(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) 