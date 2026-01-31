import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function creates the 'Pipeline' that will clean and scale the data.
        """
        try:
            # We only want to scale the numeric columns (Sensors + Derived Features)
            # We don't scale 'unit_nr' or 'RUL'
            pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")), # Handle missing values
                    ("scaler", StandardScaler()) # Standardize data (Mean=0, Std=1)
                ]
            )
            return pipeline

        except Exception as e:
            raise CustomException(e, sys)

    def add_features(self, df):
        """
        Re-creating the Rolling Means and Slopes from the Notebook.
        """
        try:
            logging.info("Engineering features (Rolling Means & Slopes)...")
            
            # 1. Define the useful sensors (The "Trenders")
            sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 
                       's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
            
            # 2. Rolling Mean & Std (Window = 5)
            for sensor in sensors:
                df[f'{sensor}_mean'] = df.groupby('unit_nr')[sensor].transform(
                    lambda x: x.rolling(window=5).mean())
                df[f'{sensor}_std'] = df.groupby('unit_nr')[sensor].transform(
                    lambda x: x.rolling(window=5).std())

            # 3. Slope (Lag = 5)
            # Sensors that showed strong trends
            sensors_slope = ['s_2', 's_3', 's_4', 's_7', 's_11', 's_12', 's_15', 's_17', 's_20', 's_21']
            for sensor in sensors_slope:
                df[f'{sensor}_slope'] = df.groupby('unit_nr')[sensor].diff(periods=5)
            
            # 4. Fill NaNs created by rolling/diff with 0
            df.fillna(0, inplace=True)
            
            return df
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Read train and test data completed")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Obtaining preprocessing object")

            # --- STEP 1: FEATURE ENGINEERING ---
            train_df = self.add_features(train_df)
            test_df = self.add_features(test_df)
            
            # --- STEP 2: DEFINE INPUTS (X) AND TARGET (Y) ---
            target_column_name = "RUL"
            drop_columns = [target_column_name, "unit_nr", "time_cycles"]

            input_feature_train_df = train_df.drop(columns=drop_columns)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns)
            target_feature_test_df = test_df[target_column_name]

            # --- STEP 3: APPLY SCALING ---
            preprocessing_obj = self.get_data_transformer_object()
            
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            # Fit on TRAIN, Transform on TEST
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine X and y back into a single array for the next step
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # --- STEP 4: SAVE THE PREPROCESSOR ---
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info(f"Saved preprocessing object.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e, sys)