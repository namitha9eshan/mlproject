import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomerException
from src.logger import logging
from src.utils import save_object
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path : str  = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        '''
        This function is responsible for data transformation 
        based on the different types of data.
        '''
        try:
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = [
                    'gender',
                    'race_ethnicity',                 
                    'parental_level_of_education',    
                    'lunch',                         
                    'test_preparation_course',
            ]

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),#handle the missing values 
                    ('scaler', StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),#Since these are categorical data, 
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info('Numerical columns standard scalling completed')
            logging.info('Categorical columns encoding completed')

            #combined the numerical_pipeline with the categorical_pipeline 
            logging.info(f'numerical_columns : {numerical_columns}')
            logging.info(f'numerical_columns : {categorical_columns}')

            preprocessor = ColumnTransformer(
                [('numerical_pipeline',numerical_pipeline, numerical_columns),#('name of the pipeline', actual variable, what columns that it needs to be applied to)
                 ('categorical_pipeline',categorical_pipeline,categorical_columns)
                ]                
            )

            return preprocessor
        except Exception as e:
            raise CustomerException(e, sys)

    def initiate_data_transformation(self,train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info('Read train and test data completed')

            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformer_obj()

            target_column_name = "math_score"
            numerical_columns = ['reading_score', 'writing_score']

            input_feature_train_df = train_df.drop(columns= [target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis= 1)
            target_feature_test_df = test_df[target_column_name]


            logging.info(f'Applying preprocessing objects on training dataframe and testing dataframe')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info('Saved preprocessing object')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
       
        except Exception as e:
            raise CustomerException(e,sys)
