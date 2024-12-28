# import io
import requests
# from PIL import Image
import streamlit as st

# interact with FastAPI endpoint

# файл отправлять как file: FormData
backend = "http://speechgpt:8000/predict"


def process(server_url: str):

    r = requests.post(
        server_url, timeout=8000
    )

    return r


if st.button("Get segmentation map"):
    segments = process(backend)
    if segments:
        st.write(segments)  # Вывод результата
    else:
        st.write("Не удалось получить результат.")
