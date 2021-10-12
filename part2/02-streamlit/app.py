import streamlit as st

import io
import os
import yaml

from PIL import Image

from predict import load_model, get_prediction

from confirm_button_hack import cache_on_button_press

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")


root_password = 'password'


def main():
    st.title("Mask Classification Model")
    #TODO뭔가 지저분한..
    asset_dir = "../../assets/mask_task/"

    with open(os.path.join(asset_dir,"config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model_path = os.path.join(asset_dir,config['model_name'])

    model = load_model(model_path)
    model.eval()

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg","png"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))

        st.image(image, caption='Uploaded Image')
        st.write("Classifying...")
        _, y_hat = get_prediction(model, image_bytes)
        label = config['classes'][y_hat.item()]

        st.write(f'label is {label}')


@cache_on_button_press('Authenticate')
def authenticate(password):
    return password == root_password


password = st.text_input('password', type="password")

if authenticate(password):
    st.success('You are authenticated!')
    main()
else:
    st.error('The password is invalid.')