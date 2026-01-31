import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

# 1. Configuration: Where to save the output files
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # --- STEP A: LOAD RAW DATA ---
            # We are reading the same RAW text file we used in the notebook
            # Make sure this path is correct relative to where you run the script!
            raw_file_path = 'data/raw/train_FD001.txt'
            logging.info(f"Reading raw data from {raw_file_path}")
            
            # Define columns (Same logic as Notebook)
            index_names = ['unit_nr', 'time_cycles']
            setting_names = ['setting_1', 'setting_2', 'setting_3']
            sensor_names = ['s_{}'.format(i) for i in range(1, 22)] 
            col_names = index_names + setting_names + sensor_names
            
            df = pd.read_csv(raw_file_path, sep=r'\s+', header=None, names=col_names)
            logging.info('Read the dataset as dataframe')

            # --- STEP B: CALCULATE RUL (Target) ---
            # This is the "Business Logic" we prototyped in the notebook
            logging.info("Calculating RUL for training data...")
            max_cycles = df.groupby('unit_nr')['time_cycles'].max().reset_index()
            max_cycles.columns = ['unit_nr', 'max_cycle']
            df = df.merge(max_cycles, on='unit_nr', how='left')
            df['RUL'] = df['max_cycle'] - df['time_cycles']
            df.drop('max_cycle', axis=1, inplace=True)
            
            # --- STEP C: SAVE ARTIFACTS ---
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save a copy of the full raw data (with RUL)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated (Splitting by Engine ID)")
            
            # --- STEP D: TRAIN/TEST SPLIT (By Engine ID) ---
            # We don't use standard train_test_split because of data leakage
            unique_units = df['unit_nr'].unique()
            np.random.seed(42)
            np.random.shuffle(unique_units)
            
            split_point = int(len(unique_units) * 0.8) # 80% Train
            train_units = unique_units[:split_point]
            test_units = unique_units[split_point:]
            
            train_set = df[df['unit_nr'].isin(train_units)]
            test_set = df[df['unit_nr'].isin(test_units)]

            # Save the split files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

# --- TEST BLOCK (To run this file independently) ---
if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print(f"Data Ingestion Complete. Train: {train_data}, Test: {test_data}")