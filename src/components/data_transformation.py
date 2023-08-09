import sys
import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.utils import save_object

@dataclass
class DataTransformerConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformer:
    def __init__(self):
        self.data_transformer_config = DataTransformerConfig()

    def get_data_transformer_object(self):
        '''
        This function is for data transformation
        '''

        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education',
                               'lunch','test_preparation_course']
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            logging.info('numerical columns standard scaling completed')

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False)),
                ]
            )
            logging.info('Categorical columns standard scaling completed')
            
            logging.info('Column Transformer started')
            preprocessor = ColumnTransformer(
                 [                    
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,categorical_columns)
                 ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
            try:
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)

                logging.info('reading of train test data complete')
                logging.info('obtaining preprocessing object')

                preprocessing_obj = self.get_data_transformer_object()
                target_column_name = 'math_score'
                numerical_columns = ['writing_score', 'reading_score']

                input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
                target_feature_train_df = train_df[target_column_name]

                input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
                target_feature_test_df = test_df[target_column_name]

                logging.info('data transformation on train and test data started')
                input_feature_train_df_arr = preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_df_arr = preprocessing_obj.fit_transform(input_feature_test_df)

                train_arr = np.c_[
                     input_feature_train_df_arr,np.array(target_feature_train_df)
                ]

                test_arr = np.c_[
                    input_feature_test_df_arr,np.array(target_feature_test_df)
                ]
                
                logging.info('data transformation on train and test data applied')
                
                # save_object is defined in src/utils.py
                save_object(
                     file_path = self.data_transformer_config.preprocessor_obj_file_path,
                     obj = preprocessing_obj
                )
                logging.info('saved pickle file after data transformation')

                return (train_arr,
                        test_arr,
                        self.data_transformer_config.preprocessor_obj_file_path)



            except Exception as e:
                 raise CustomException(e,sys)
