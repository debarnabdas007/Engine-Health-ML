import sys
import pandas as pd
import os
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            logging.info("Loading model and preprocessor...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            logging.info("Scaling input data...")
            data_scaled = preprocessor.transform(features)
            
            logging.info("Making prediction...")
            preds = model.predict(data_scaled)
            
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, 
                 s_2: float, s_3: float, s_4: float, 
                 s_7: float, s_11: float, s_12: float, 
                 s_15: float, s_17: float, s_20: float, s_21: float):
        
        self.s_2 = s_2
        self.s_3 = s_3
        self.s_4 = s_4
        self.s_7 = s_7
        self.s_11 = s_11
        self.s_12 = s_12
        self.s_15 = s_15
        self.s_17 = s_17
        self.s_20 = s_20
        self.s_21 = s_21

    def get_data_as_dataframe(self):
        try:
            # 1. Create a dictionary with ALL possible keys
            custom_data_input_dict = {
                # Raw Sensors
                "s_2": [self.s_2], "s_3": [self.s_3], "s_4": [self.s_4],
                "s_7": [self.s_7], "s_11": [self.s_11], "s_12": [self.s_12],
                "s_15": [self.s_15], "s_17": [self.s_17], "s_20": [self.s_20], 
                "s_21": [self.s_21],

                # Missing Sensors & Settings (Fill with 0)
                "s_1": [0], "s_5": [0], "s_6": [0], "s_8": [0], "s_9": [0], 
                "s_10": [0], "s_13": [0], "s_14": [0], "s_16": [0], 
                "s_18": [0], "s_19": [0],
                "setting_1": [0], "setting_2": [0], "setting_3": [0],
                
                # Rolling Features (Fill with 0)
                "s_2_mean": [0], "s_3_mean": [0], "s_4_mean": [0], "s_7_mean": [0],
                "s_8_mean": [0], "s_9_mean": [0], "s_11_mean": [0], "s_12_mean": [0],
                "s_13_mean": [0], "s_14_mean": [0], "s_15_mean": [0], "s_17_mean": [0],
                "s_20_mean": [0], "s_21_mean": [0],

                "s_2_std": [0], "s_3_std": [0], "s_4_std": [0], "s_7_std": [0],
                "s_8_std": [0], "s_9_std": [0], "s_11_std": [0], "s_12_std": [0],
                "s_13_std": [0], "s_14_std": [0], "s_15_std": [0], "s_17_std": [0],
                "s_20_std": [0], "s_21_std": [0],

                # Slopes (Fill with 0)
                "s_2_slope": [0], "s_3_slope": [0], "s_4_slope": [0],
                "s_7_slope": [0], "s_11_slope": [0], "s_12_slope": [0],
                "s_15_slope": [0], "s_17_slope": [0], "s_20_slope": [0], 
                "s_21_slope": [0]
            }

            df = pd.DataFrame(custom_data_input_dict)

            # --- CRITICAL FIX: FORCE CORRECT COLUMN ORDER ---
            # We must recreate the exact order defined in data_transformation.py
            
            # 1. Base Columns: Settings + s_1 to s_21
            ordered_cols = ['setting_1', 'setting_2', 'setting_3'] + [f's_{i}' for i in range(1, 22)]
            
            # 2. Rolling Mean & Std (In the order the loop created them)
            sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
            for s in sensors:
                ordered_cols.append(f'{s}_mean')
                ordered_cols.append(f'{s}_std')
            
            # 3. Slopes
            sensors_slope = ['s_2', 's_3', 's_4', 's_7', 's_11', 's_12', 's_15', 's_17', 's_20', 's_21']
            for s in sensors_slope:
                ordered_cols.append(f'{s}_slope')

            # Reorder the DataFrame to match the Training Data exactly
            df = df[ordered_cols]

            return df

        except Exception as e:
            raise CustomException(e, sys)