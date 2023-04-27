import streamlit as st
import io
import yaml
from PIL import Image
from predict import load_model, get_prediction

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

# password
root_password = 'password'

def main():
    st.title("Mask Classification Model")

    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    model = load_model()
    model.eval()

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption='Uploaded Image')
        st.write("Classifying...")
        _, y_hat = get_prediction(model, image_bytes)
        label = config['classes'][y_hat.item()]

        st.write(f'label is {label}')

def authenticate(password) -> bool:
    return password == root_password

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    password = st.text_input('password', type="password")

    with st.form(key='my_form'):
        submit_button = st.form_submit_button(label='Authenticate')
        if authenticate(password):
            st.success('You are authenticated!')
            st.session_state.authenticated = True
        else:
            st.error('The password is invalid.')
else:
    main()
