import os 
import sys 
from dataclasses import dataclass
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model, load_file

# algoritma 
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# Define path save model file
@dataclass
class ModelTrainerConfig():
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


# create class for initiate model
class ModelTrainer():
    def __init__(self):
        self.model_trained_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            # Split data input and target
            logging.info("Split data input and target")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            logging.info("Split data input and target has sucsses")

            # Define dictionary of models that want to train
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Define params for hyperparameter tunning
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            # train and evaluate model
            logging.info("initiate model trainer")
            model_reports:dict = evaluate_model(X_train=X_train,y_train=y_train,
                                                X_test=X_test, y_test=y_test,
                                                models=models, params=params)
            logging.info("Model has trained")

            # Get Best model
            # sorted score
            best_model_score = max(sorted(model_reports.values()))
            # best model name
            best_model_name = list(model_reports.keys())[list(model_reports.values()).index(best_model_score)]
            # best model
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("Model R2 score less than 60%")
            logging.info(f"Best found model on both training and testing dataset")

            # save model
            save_object(file_path=self.model_trained_config.trained_model_file_path,
                        object=best_model)
            
            # predic 
            prediction = best_model.predict(X_test)
            score = r2_score(y_test, prediction)

            return score
                                            
        except Exception as error:
            raise CustomException(error, sys)