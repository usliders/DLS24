# config.py
import os

API_TOKEN = "6..."
UPLOADS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".", "data", "uploads"))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".", "data", "results"))
PROMTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".", "data", "promts"))
LOG_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "bot_log.log"))
encoded_image = None
IMAGE_SIZE = (512, 512)
API_HTTPS = "https://6702bbc217bef80f74.gradio.live"
TIME_ASK = 150
GRADIO_STATIC_PATH = "D:/FooocusControl/Fooocus/outputs/gradio"

