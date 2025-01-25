# api.py
from gradio_client import Client

def get_api_client():
    return Client("http://127.0.0.1:7860/, serialize=False")
