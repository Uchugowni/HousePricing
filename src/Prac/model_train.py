import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from src.utils import save_object
from src.logger import logging

#"lr", "DT", "KNN", "RF", "AB", "XGB", "GB", "CatB"

class model_data_store:
    model_store = os.path.join("artifacts", "modelp.pkl")


class model_train_data:
    def __init__(self) -> None:
        self.data_store = model_data_store()
        
    def model_training(self, train_arr, test_arr):
        try:
            X_train, Y_train, X_test, Y_test = (train_arr[:,:-1], train_arr[:,-1], test_arr[:,:-1], test_arr[:,-1])
            logging.info("Array train and test data splited into as model train_test_split format")
            Models = {
                "lr" : LinearRegression(),
                "DT" : DecisionTreeRegressor(),
                "KNN" : KNeighborsRegressor(),
                "RF" : RandomForestRegressor(),
                "AB" : AdaBoostRegressor(),
                "XGB" : XGBRegressor(),
                "GB" : GradientBoostingRegressor(),
                "CatB" : CatBoostRegressor()
            }
            logging.info("Required models are stored in Model tuple")
            report = {}
            for i in range(len(Models)):
                model = list(Models.values())[i]
                model.fit(X_train, Y_train)
                Y_test_pred = model.predict(X_test)
                r2score = r2_score(Y_test, Y_test_pred)
                report[list(Models.keys())[i]] = r2score
            print(report)
            print(list(report.values()))

            best_model_score = max(sorted(report.values()))
            if best_model_score<0.8:
                raise CustomException("No Best model found")

            best_model_name = list(report.keys())[list(report.values()).index(best_model_score)]
            best_model = Models[best_model_name]
            logging.info(f"This model is best model {best_model} and {best_model_score} in this dataset")

            save_object(
                file_path = self.data_store.model_store,
                obj=best_model
            )

            print(best_model, best_model_score)

            Predicted = best_model.predict(X_test)
            print(r2_score(Y_test, Predicted))


        except Exception as e:
            raise CustomException(e, sys)