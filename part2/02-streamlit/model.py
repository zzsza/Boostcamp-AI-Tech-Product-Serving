import torch
from transformers import ElectraModel

class MySTSModel(ElectraModel):
    pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.plm(x)['logits']
        return x
