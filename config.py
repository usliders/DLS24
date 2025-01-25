# config.py
import os

API_TOKEN = "6833480462:AAFNbb2pan_yesaPU1JxU9SE3fjCLcPPPQU"
UPLOADS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".", "data", "uploads"))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".", "data", "results"))
PROMTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".", "data", "promts"))
LOG_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "bot_log.log"))
encoded_image = None
IMAGE_SIZE = (512, 512)


