from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    # 1. Run Data Ingestion (Raw -> Train/Test CSVs)
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # 2. Run Data Transformation (CSVs -> Arrays + Preprocessor.pkl)
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # 3. Run Model Trainer (Arrays -> Model.pkl)
    model_trainer = ModelTrainer()
    print("Training Model...")
    model_trainer.initiate_model_trainer(train_arr, test_arr)