import os
import sys

import numpy as np
import pandas as pd

from src.exception import CustomException

def save_object(obj, file_path):
    """
    Save an object to a file using pickle.
    """
    import pickle
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  

        # Save the object to the specified file path
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        print(f"Object saved to {file_path}")
        
    except Exception as e:
        raise CustomException(e, sys)