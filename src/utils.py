import os
import sys
import pickle
import numpy as np
import pandas as pd 
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        logging.info('Exception Occured while save object')
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        logging.info('Exception Occured while load object')
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
        logging.info('Exception Occured while evaluating model')
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
        logging.info('Exception Occured while evaluating model')
        raise CustomException(e, sys)
    

def model_metrics(Actual, predicted):
    try :
        mae = mean_absolute_error(Actual, predicted)
        mse = mean_squared_error(Actual, predicted)
        rmse = np.sqrt(mse)
        r2_square = r2_score(Actual, predicted)
        return mse, mae, rmse, r2_square
    except Exception as e:
        logging.info('Exception Occured while evaluating metric')
        raise CustomException(e,sys)