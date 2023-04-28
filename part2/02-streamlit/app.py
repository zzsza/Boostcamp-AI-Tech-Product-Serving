import streamlit as st

import io
import os
import yaml

from PIL import Image

from predict import load_model, get_prediction


# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")


# main
def main():
    st.title("Mask Classification Model")

    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = load_model()
    model.eval()

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg","png"])

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption='Uploaded Image')
        st.write("Classifying...")
        _, y_hat = get_prediction(model, image_bytes)
        label = config['classes'][y_hat.item()]

        st.write(f'label is {label}')


# password
root_password = 'password'

# password가 맞는지 확인하는 함수 정의
def authenticate(password) -> bool:
    return password == root_password

# session state로 정의해 global 변수 처리 (streamlit 특성 상 계속 갱신되니 고정하는 개념)
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# password 입력 후 main 함수 실행 
if not st.session_state.authenticated:                      # 처음엔 authenticated = false임
    password = st.text_input('password', type="password")   # password 입력

    with st.form(key='my_form'): # form 정의 
        submit_button = st.form_submit_button(label='Authenticate') # Authenticate라는 버튼 생성
        if authenticate(password):                                  # 만약 password가 정확하다면 
            st.success('You are authenticated!')
            st.session_state.authenticated = True                   # authenticated가 true가 되고 갱신
        else:
            st.error('The password is invalid.')
else:
    main() # 갱신되었으니 main 함수가 실행된다. 