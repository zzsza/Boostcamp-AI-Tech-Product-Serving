import streamlit as st
import yaml
from predict import load_model, get_prediction
import io
from PIL import Image
from confirm_button_hack import cache_on_button_press

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

st.write("Hello World!")

# TODO: Streamlit App 만들기
# TODO: Voila 코드 리펙토링(app.py, model.py, predict.py, utils.py)
def main():
    st.title("Mask Classification Model")
    
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    model = load_model()
    model.eval()

    # TODO: File Uploader 구현
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    uploaded_file
    
    if uploaded_file:
        # TODO: 이미지 View
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption="Uploaded Image")
        st.write("Classifying...")
        # TODO: 예측
        _, y_hat = get_prediction(model, image_bytes)
        # TODO: 예측 결과 출력
        label = config['classes'][y_hat.item()]
        st.write(f"Prediction Reponse is {label}")
        
    

# TODO: 암호 입력
root_password = 'password'

password = st.text_input('password', type='password')

@cache_on_button_press('Authenticate')
def authenticate(password) -> bool:
    st.write(type(password))
    return password == root_password

if authenticate(password):
    st.success("You are authenticated!")
    main()
else :
    st.error("The password is invalid.")