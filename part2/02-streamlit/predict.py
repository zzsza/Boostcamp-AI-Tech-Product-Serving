import torch
import streamlit as st
from model import MaskClassificationModel
from utils import transform_image

@st.cache
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MaskClassificationModel(num_classes=18).to(device)
    return model


def get_prediction(model, image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return tensor, y_hat
