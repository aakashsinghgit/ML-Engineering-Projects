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
    
def load_object(file_path):
    """
    Load an object from a file using pickle.
    """
    import pickle
    try:
        # Load the object from the specified file path
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        # print(f"Object loaded from {file_path}")
        return obj
        
    except Exception as e:
        raise CustomException(e, sys)