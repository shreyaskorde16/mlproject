import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from dataclasses import dataclass
import sys
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

'''
With @dataclass, you don't need to write explicit __init__ methods to initialize your 
class attributes. The __init__ method is automatically generated based on the class attributes.
'''
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')
    
    
class DataTransformation:
    def __init__(self):
        self.data_trransformation_config=DataTransformationConfig()
        
    def get_data_transformer_object(self):
        
        '''
        This function is responsible for data transformation
        '''
        
        try:
            numerical_columns = ["writing_score", "reading_scorer"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(statergy="median"))
                    ("scaler", StandardScaler())
                ]                
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent"))
                    ("one_hot_encoder", OneHotEncoder())
                    ("scaler", StandardScaler())
                ]
            )
            logging.info(f"Numerical columns are {numerical_columns}")
            logging.info(f"Categorical columns are {categorical_columns}")
            
            preprocessor = ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, numerical_columns)
                ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
        