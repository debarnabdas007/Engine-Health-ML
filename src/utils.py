import os
import sys
import pandas as pd
import joblib
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Standard function to save any Python object (Model, Preprocessor) as a pickle file.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        joblib.dump(obj, file_path)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Standard function to load a pickle file.
    """
    try:
        return joblib.load(file_path)

    except Exception as e:
        raise CustomException(e, sys)