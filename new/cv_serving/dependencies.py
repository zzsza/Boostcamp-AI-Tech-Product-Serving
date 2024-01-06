import torch
from loguru import logger

from model.model import BaseModel

model = None


def load_model(model_path: str) -> None:
    global model
    logger.info(f"Loading model from {model_path}.")

    model = BaseModel(num_classes=18)
    device = torch.device("cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))

    logger.info("Model loaded.")


def get_model() -> BaseModel:
    global model
    return model

