import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.utils import save_object, evaluate_model, evaluate_models
from sklearn.metrics import r2_score
from src.exception import CustomException
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model_p.pkl")


class ModelTrain_Process:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def predict_train_test(self, train_arr, test_arr):
        try:
            X_train, Y_train, X_test, Y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1] )
            Models = {
                "lr": LinearRegression(),
                "DT": DecisionTreeRegressor(),
                "KNN": KNeighborsRegressor(),
                "RF": RandomForestRegressor(),
                "GD": GradientBoostingRegressor(),
                "XGB": XGBRegressor(),
                "AB": AdaBoostRegressor(),
                "CB": CatBoostRegressor()
            }

            Params = {
                "lr" : {
                    },
                "DT" : {
                    'criterion' : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_depth' : [3, 6, 9, 12],
                    'min_samples_leaf' : [3, 8, 12, 15]
                    },
                'KNN': {
                    'n_neighbors': [3, 6, 9, 18]
                    },
                'RF': {
                    'n_estimators' : [3, 6, 9, 18],
                    'max_depth' :[3, 6, 9, 12]
                    },
                'GD': {
                    'learning_rate' : [.1, .01, .05, .001],
                    'n_estimators' : [3, 6, 9, 12],
                    'max_depth' : [3, 6, 9, 12]
                },
                'XGB' : {
                    'learning_rate' : [.1, .01, .05, .001],
                    'n_estimators' : [3, 6, 9, 12]
                },
                'AB' : {
                    'learning_rate' : [.1, .01, .05, .001],
                    'n_estimators' : [3, 6, 9, 12]
                },
                'CB' : {
                    'depth' : [3, 6, 9, 12],
                    'learning_rate' : [.01, .05, .1],
                    'iterations' : [3, 6, 9, 12]
                }
            }

            report: dict = evaluate_model(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test, models=Models, param=Params)
            
            best_model_score = max(sorted(report.values()))
            best_model_name = list(report.keys())[list(report.values()).index(best_model_score)]
            best_model = Models[best_model_name]
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            r2value = r2_score(Y_test, predicted)
            return r2value
        except Exception as e:
            raise CustomException(e, sys)

        

