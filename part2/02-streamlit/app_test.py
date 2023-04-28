import streamlit as st

import io
import os
import yaml

from PIL import Image

from predict import load_model, get_prediction

from confirm_button_hack import cache_on_button_press

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

st.write("Hello World!")

def main():
    st.title("Mask Classification Model")

    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = load_model()
    model.eval()

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg","png"])
    uploaded_file
    
    # TODO: File Uploader
    # TODO: 이미지 View
    # TODO: 예측
    # TODO: 예측 결과 출력
