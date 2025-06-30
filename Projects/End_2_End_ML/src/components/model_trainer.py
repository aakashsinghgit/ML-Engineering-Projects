import os
import sys  
from dataclasses import dataclass   

from src.exception import CustomException
from src.logger import logging  
from src.utils import save_object   

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()    

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], 
                train_array[:, -1], 
                test_array[:, :-1], 
                test_array[:, -1]
            )

            models = {
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "LinearRegression": LinearRegression(),
                "SVR": SVR(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=0),
                "XGBRegressor": XGBRegressor()
            }

            # Hyperparameters can be tuned here if needed

            logging.info("Starting model training")
            model_report = {}
            for model_name, model in models.items():
                logging.info(f"Training {model_name}")
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                
                model_report[model_name] = {
                    "r2_score": r2,
                    "mean_absolute_error": mae,
                    "mean_squared_error": mse
                }
                
            best_model_name = max(model_report, key=lambda x: model_report[x]['r2_score'])
            best_model = models[best_model_name]
            
            logging.info(f"Best model found: {best_model_name} with R2 score: {model_report[best_model_name]['r2_score']}")
            
            save_object(best_model, self.model_trainer_config.trained_model_file_path)
            
            return best_model_name, model_report
        
        except Exception as e:
            raise CustomException(e, sys)