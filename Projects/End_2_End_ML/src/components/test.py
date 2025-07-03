## This file can be used to test the data ingestion, transformation, and model training components of the  training pipeline.
## Later on, we will create a separate training pipeline file for this

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_ingestion import DataIngestionConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig 

if __name__ == "__main__":
# Data Ingestion
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
    print("Data ingestion completed successfully.")
    
    # Data Transformation
    data_transformation = DataTransformation()
    train_array, test_array = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    print("Data transformation completed successfully.")   

    # Model Trainer
    model_trainer = ModelTrainer()
    Best_model_name, model_report = model_trainer.initiate_model_trainer(train_array, test_array)
    print(f"Best model: {Best_model_name}")
    print("Model training completed successfully.")
    # You should see the model.pkl file in artifacts folder and the preprocessor.pkl file in artifacts folder. This is the final output of the training pipeline.
