# config.py
import os

API_TOKEN = "xxx"
UPLOADS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".", "data", "uploads"))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".", "data", "results"))
PROMTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".", "data", "promts"))
LOG_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "bot_log.log"))
encoded_image = None
IMAGE_SIZE = (512, 512)


