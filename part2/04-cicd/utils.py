import io
import numpy as np
from PIL import Image

import albumentations
import albumentations.pytorch
import torch


def transform_image(image_bytes: bytes) -> torch.Tensor:
    transform = albumentations.Compose([
            albumentations.Resize(height=512, width=384),
            albumentations.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.2, 0.2, 0.2)),
            albumentations.pytorch.transforms.ToTensorV2()
        ])
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    image_array = np.array(image)
    return transform(image=image_array)['image'].unsqueeze(0)