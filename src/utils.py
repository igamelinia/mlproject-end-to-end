import os
import sys
import pickle
import pandas as pd 
import numpy as np 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.exception import CustomException

# Create function for save object
def save_object(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(object, file_obj)

    except Exception as error:
        raise CustomException(error, sys)
    
# Create funstion evaluate model
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        result={}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            # train model with cross validation
            gs = GridSearchCV(model, param, cv=5, scoring="r2")
            gs.fit(X_train, y_train)

            # train model with best params
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # prediction train and test
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            # model r2 score
            train_score = r2_score(y_train, train_pred)
            test_score = r2_score(y_test, test_pred)

            result[list(models.keys())[i]] = test_score

        return result

    except Exception as error:
        raise CustomException(error, sys)
    
# function for load file
def load_file(file_path):
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)

    except Exception as error:
        raise CustomException(error, sys)