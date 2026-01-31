import os
import sys
from dataclasses import dataclass

from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            
            # The arrays come in as [X, y] combined. We need to slice them.
            # :-1 means "All columns except the last one" (Features)
            # -1 means "Only the last column" (Target/RUL)
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Initialize XGBoost with the best params from our Notebook
            model = XGBRegressor(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=6, 
                n_jobs=-1,
                random_state=42
            )

            logging.info("Training XGBoost Model...")
            model.fit(X_train, y_train)

            logging.info("Model trained successfully")

            # Evaluate
            predicted = model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            rmse = np.sqrt(mean_squared_error(y_test, predicted))

            print(f"Model Performance -> R2: {r2:.4f}, RMSE: {rmse:.4f}")
            logging.info(f"Model Performance -> R2: {r2} RMSE: {rmse}")

            # Save the model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            return r2

        except Exception as e:
            raise CustomException(e, sys)