import logging
import os
from datetime import datetime


# make file name log
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# define path for folder log
folder_path = os.path.join(os.getcwd(), "logs")

# make folder log
os.makedirs(folder_path, exist_ok=True)

# define final file log path
LOG_FILE_PATH = os.path.join(folder_path, LOG_FILE)

# Mengonfigurasi Logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# # code for test code
# if __name__=="__main__":
#     logging.info("Logging has started")