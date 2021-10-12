import torch
import streamlit as st
from model import MyEfficientNet
from utils import transform_image


@st.cache
def load_model(model_path=""):
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MyEfficientNet(num_classes=18).to(device)


    if str(device) =="cpu":
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_path))
        model.to(device)

    return model


def get_prediction(model, image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return tensor, y_hat
