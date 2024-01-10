# Custom Model Template
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        """
        모델 아키텍쳐를 직접 작성합니다
        """

    def forward(self, x):
        """
        위 모델에 대한 forward propagation
        """
        return x
