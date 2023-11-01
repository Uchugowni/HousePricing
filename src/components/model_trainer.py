import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor) 
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split traing and test data")
            X_train, Y_train, X_test, Y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )
            Models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gredient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbors": KNeighborsRegressor(),
                "XGB Regression": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor()
            }

            model_report: dict = evaluate_models(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test, models=Models)

            print(model_report)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = Models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No Best model found")
            logging.info(f"Best found model {best_model} on both traing and test dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(Y_test, predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)