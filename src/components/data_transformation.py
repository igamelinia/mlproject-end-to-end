import sys
import os
import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# Define file path
class DataTransformConfig():
    processor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


# Create class for initiate
class DataTransformation():
    def __init__(self):
        self.data_transformation_config=DataTransformConfig()

    def get_data_transform_object(self):
        try:
            # Devide columns by transformation
            num_columns = ["writing_score", "reading_score"]
            cat_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Define Numeric Pipeline
            num_pipeline = Pipeline(
                steps=[("num_imputer", SimpleImputer(strategy="median")),
                       ("Scaler", StandardScaler())
                       ]
            )

            # Define Categoric Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("cat_imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Columns numeric : {num_columns}")
            logging.info(f"Columns categoric : {cat_columns}")

            # Combine Pipeline
            preprocessor = ColumnTransformer([
                ("numeric_pipeline", num_pipeline, num_columns),
                ("categoric_pipepline", cat_pipeline, cat_columns)
            ])

            return preprocessor

        except Exception as error:
            raise CustomException(error, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read data train and test
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read data Train and Test")

            # Load preprocessor
            logging.info("Obtaining Preprocessor")
            preprocessing_obj = self.get_data_transform_object()

            # Define variable dependent
            target_column = "math_score"
            num_columns = ["writing_score", "reading_score"]

            # split input and target feature
            input_features_train = train_df.drop(columns=[target_column], axis=1)
            target_features_train = train_df[target_column]

            input_features_test = test_df.drop(columns=[target_column], axis=1)
            target_features_test = test_df[target_column]

            # Fit and transform Preprocessing
            logging.info("Applying preprocessing object on training dataframe and testing dataframe")
            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train)
            input_features_test_arr = preprocessing_obj.transform(input_features_test)

            # Combine array of input features and target feature
            train_arr = np.c_[input_features_train_arr, np.array(target_features_train)]
            test_arr = np.c_[input_features_test_arr, np.array(target_features_test)]

            # Save prepocessor as pkl file
            save_object(file_path=self.data_transformation_config.processor_obj_file_path, object=preprocessing_obj)
            logging.info("Save preprocessor as pickle file has sucsses")

            return (train_arr, 
                    test_arr,
                    self.data_transformation_config.processor_obj_file_path )

        except Exception as error:
            raise CustomException(error, sys)
