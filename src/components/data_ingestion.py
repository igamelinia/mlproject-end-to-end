import os
import sys 
import pandas as pd 
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException 


# Class  for define file path 
@dataclass
class DataIngestionConfig():
    train_file_path:str = os.path.join("artifacts", "train.csv")
    test_file_path:str = os.path.join("artifacts", "test.csv")
    raw_file_path:str = os.path.join("artifacts", "data.csv")

# Class for ingestion prosses (read and prepare data before preprocesing)
class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try :
            # Read dataset from csv
            df = pd.read_csv("notebook\data\stud.csv")
            logging.info("Read dataset as Dataframe")

            # Create folder artifacts
            os.makedirs(os.path.dirname(self.ingestion_config.train_file_path), exist_ok=True)

            # Save dataset as file csv 
            df.to_csv(self.ingestion_config.raw_file_path, index=False, header=True)

            # Split data to Train and test
            logging.info("Split dataset to Train and Test")
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and Test data
            train_data.to_csv(self.ingestion_config.train_file_path, index=False, header=True)
            test_data.to_csv(self.ingestion_config.test_file_path, index=False, header=True)
            logging.info("Process Data Ingestion has complated")

            return (
                self.ingestion_config.train_file_path,
                self.ingestion_config.test_file_path
            )

        except Exception as error:
            raise CustomException(error, sys)
        

# Code to check code
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()
        






