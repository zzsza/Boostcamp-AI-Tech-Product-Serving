import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


def get_model(model_path: str = "../../assets/mask_task/model.pth"):
    """Model을 가져옵니다
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyEfficientNet(num_classes=18).to(device)
    if str(device) =="cpu":
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_path))
        model.to(device)
    model.eval()
    return model


class MyEfficientNet(nn.Module) :
    '''
    EfiicientNet-b4의 출력층만 변경합니다.
    한번에 18개의 Class를 예측하는 형태의 Model입니다.
    '''
    def __init__(self, num_classes: int = 1000) :
        super(MyEfficientNet, self).__init__()
        self.EFF = EfficientNet.from_pretrained('efficientnet-b4', in_channels=3, num_classes=num_classes)
    
    def forward(self, x) -> torch.Tensor:
        x = self.EFF(x)
        x = F.softmax(x, dim=1)
        return x

