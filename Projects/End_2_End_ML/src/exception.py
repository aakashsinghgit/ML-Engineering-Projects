import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    """
    Returns a detailed error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = "Error occurred in script: [{file_ name}] at line number: [{line_number}] with message: [{str(error)}]"
    

class CustomException(Exception):
    """
    Custom exception class to handle exceptions with detailed error messages.
    """
    def __init__(self, error, error_detail: sys):
        super().__init__(error_message_detail(error, error_detail))
        self.error_message = error_message_detail(error, error_detail)

        def __str__(self):
            return self.error_message
    
# if __name__ == "__main__":
#     try:
#         a = 1 / 0  # Example to raise an exception
#     except Exception as e:
#         logging.info("Divide by zero exception occurred")
#         raise CustomException(e, sys) from e

        