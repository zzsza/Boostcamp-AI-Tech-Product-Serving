import albumentations
import albumentations.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from bentoml import BentoService, api, env, artifacts
from bentoml.adapters import JsonOutput, ImageInput
from bentoml.artifact import PytorchModelArtifact
from efficientnet_pytorch import EfficientNet
from imageio.core.util import Array

classes = {
    0: ["Wear", "Male", "under 30"],
    1: ["Wear", "Male", "between 30 and 60"],
    2: ["Wear", "Male", "over 60"],
    3: ["Wear", "Female", "under 30"],
    4: ["Wear", "Female", "between 30 and 60"],
    5: ["Wear", "Female", "over 60"],
    6: ["Incorrect", "Male", "under 30"],
    7: ["Incorrect", "Male", "between 30 and 60"],
    8: ["Incorrect", "Male", "over 60"],
    9: ["Incorrect", "Female", "under 30"],
    10: ["Incorrect", "Female", "between 30 and 60"],
    11: ["Incorrect", "Female", "over 60"],
    12: ["Not Wear", "Male", "under 30"],
    13: ["Not Wear", "Male", "between 30 and 60"],
    14: ["Not Wear", "Male", "over 60"],
    15: ["Not Wear", "Female", "under 30"],
    16: ["Not Wear", "Female", "between 30 and 60"],
    17: ["Not Wear", "Female", "over 60"],
}


class MyEfficientNet(nn.Module):
    """
    EfiicientNet-b4의 출력층만 변경합니다.
    한번에 18개의 Class를 예측하는 형태의 Model입니다.
    """

    def __init__(self, num_classes: int = 18):
        super(MyEfficientNet, self).__init__()
        self.EFF = EfficientNet.from_pretrained(
            "efficientnet-b4", in_channels=3, num_classes=num_classes
        )

    def forward(self, x) -> torch.Tensor:
        x = self.EFF(x)
        x = F.softmax(x, dim=1)
        return x


@env(infer_pip_packages=True)
@artifacts([PytorchModelArtifact("model")])
class MaskAPIService(BentoService):
    def transform(self, image_array: Array):
        _transform = albumentations.Compose(
            [
                albumentations.Resize(height=512, width=384),
                albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
                albumentations.pytorch.transforms.ToTensorV2(),
            ]
        )
        return _transform(image=image_array)["image"].unsqueeze(0)

    def get_label_from_class(self, class_: int):
        return classes[class_]

    @api(input=ImageInput(), output=JsonOutput())
    def predict(self, image_array: Array):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        transformed_image = self.transform(image_array).to(device)
        outputs = self.artifacts.model.forward(transformed_image)
        _, y_hats = outputs.max(1)
        return self.get_label_from_class(class_=y_hats.item())


if __name__ == "__main__":
    import torch

    bento_svc = MaskAPIService()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyEfficientNet().to(device)
    state_dict = torch.load(
        "../../../assets/mask_task/model.pth", map_location=device
    )
    model.load_state_dict(state_dict=state_dict)
    bento_svc.pack("model", model)
    saved_path = bento_svc.save()
    print(saved_path)
