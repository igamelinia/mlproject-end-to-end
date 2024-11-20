import sys
from src.logger import logging

# Fungsi untuk mengambil nama file, no baris dan pesan error
def error_message_detail(error, error_detail:sys):
    """
    This Function to take file name, no line, and error message
    """
    # define variable to get traceback
    _,_,exc_tb = error_detail.exc_info()
    
    # define variable file and number of error occured
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_error = exc_tb.tb_lineno

    error_message = f'Error occured in python script name [{file_name}] line number [{line_error}] error message[{str(error)}]'
    return error_message

# Create class 
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message

# # Code for check code
# if __name__=="__main__":
#     try :
#         a=1/0
#     except Exception as error:
#         logging.info("Devide by zero")
#         raise CustomException(error, sys)