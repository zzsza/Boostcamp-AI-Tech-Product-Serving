import io
import os
from pathlib import Path

import requests
import yaml
from PIL import Image

import streamlit as st

# SETTING PAGE CONFIG TO WIDE MODE
from app.confirm_button_hack import cache_on_button_press

ASSETS_DIR_PATH = os.path.join(Path(__file__).parent.parent.parent.parent, "assets")

st.set_page_config(layout="wide")

root_password = 'password'


def main():
    st.title("Mask Classification Model")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))

        st.image(image, caption='Uploaded Image')
        st.write("Classifying...")

        files = [
            ('files', (uploaded_file.name, image_bytes,
                       uploaded_file.type))
        ]
        response = requests.post("http://localhost:5000/predict", files=files)
        label = response.json()
        st.write(f'label is {label}')


@cache_on_button_press('Authenticate')
def authenticate(password) -> bool:
    print(type(password))
    return password == root_password


password = st.text_input('password', type="password")

if authenticate(password):
    st.success('You are authenticated!')
    main()
else:
    st.error('The password is invalid.')
