import io
from typing import List, Dict, Any

import albumentations
import albumentations.pytorch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from efficientnet_pytorch import EfficientNet


class MyEfficientNet(nn.Module):
    """
    EfiicientNet-b4의 출력층만 변경합니다.
    한번에 18개의 Class를 예측하는 형태의 Model입니다.
    """

    def __init__(self, num_classes: int = 18):
        super(MyEfficientNet, self).__init__()
        self.EFF = EfficientNet.from_pretrained("efficientnet-b4", in_channels=3, num_classes=num_classes)

    def forward(self, x) -> torch.Tensor:
        x = self.EFF(x)
        x = F.softmax(x, dim=1)
        return x


def get_model(model_path: str = "../../assets/mask_task/model.pth") -> MyEfficientNet:
    """Model을 가져옵니다"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyEfficientNet(num_classes=18).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def _transform_image(image_bytes: bytes):
    transform = albumentations.Compose(
        [
            albumentations.Resize(height=512, width=384),
            albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
            albumentations.pytorch.transforms.ToTensorV2(),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image_array = np.array(image)
    return transform(image=image_array)["image"].unsqueeze(0)


def predict_from_image_byte(model: MyEfficientNet, image_bytes: bytes, config: Dict[str, Any]) -> List[str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformed_image = _transform_image(image_bytes).to(device)
    outputs = model.forward(transformed_image)
    _, y_hat = outputs.max(1)
    return config["classes"][y_hat.item()]


def get_config(config_path: str = "../../assets/mask_task/config.yaml"):
    import yaml

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config
