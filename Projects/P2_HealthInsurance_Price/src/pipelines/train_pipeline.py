## Let's train the model and save the model and preprocessor to artifacts folder.

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import save_object

from src.logger import logging
from src.exception import CustomException
import sys

if __name__ == "__main__":
    try:
        # Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed successfully.")
        
        # Data Transformation
        data_transformation = DataTransformation()
        train_array, test_array = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info("Data transformation completed successfully.")     

        # Model Trainer
        model_trainer = ModelTrainer()
        best_model_name, model_report = model_trainer.initiate_model_trainer(train_array, test_array)
        logging.info(f"Best model: {best_model_name}")          
        logging.info("Model training completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise CustomException(e, sys)
    
## You should see the model.pkl file in artifacts folder and the preprocessor.pkl file in artifacts folder. This is the final output of the training pipeline.
