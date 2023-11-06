import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor) 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, model_metrics
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

            params = {
                "Random Forest" : {
                    'n_estimators':[8, 9]
                    #criterion, max_depth, min_samples_leaf
                    },
                "Decision Tree" : {
                    'criterion' : ['squared_error', 'friedman_mse']
                    #splitter, max_depth, min_samples_leaf, max_features  
                    },
                "Gredient Boosting" : {
                    #'loss':['squared_error', 'absolute_error', 'huber', 'quantile']
                    'learning_rate' : [.1, .01],
                    'subsample' : [0.6, 0.7],
                    #criterion, max_features, max_depth, min_samples_leaf
                    'n_estimators' : [8, 7]
                    },
                "Linear Regression" : {},
                "K-Neighbors": {
                    'n_neighbors' : [5, 7]
                    #weights, leaf_size, p, metric
                    },
                "XGB Regression": {
                    'learning_rate' : [.1, .01, .05],
                    'n_estimators' : [8, 6]
                    #n_estimators, random_state, n_jobs, max_depth, learning_rate
                    },
                "CatBoostRegressor": {
                    'depth' : [6, 8, 10],
                    'learning_rate' : [0.01, 0.05],
                    'iterations' : [5, 10]
                    #iterations, learning_rate, loss_function, max_depth, n_estimators, random_state, depth
                    },
                "AdaBoostRegressor": {
                    'learning_rate' : [.1, .01],
                    'n_estimators' : [8, 6]
                    #n_estimators, learning_rate, loss, random_state, 
                }
            }
            model_report: dict = evaluate_models(x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test, models=Models, param=params)

            print(model_report)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = Models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No Best model found")
            
            logging.info(f"Model name: {best_model_name}")
            logging.info(f"r2Score : {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            mse, mae, rmse, r2_square = model_metrics(Y_test, predicted)
            return mse, mae, rmse, r2_square
        
        except Exception as e:
            logging.info('Exception Occured while model training')
            raise CustomException(e, sys)