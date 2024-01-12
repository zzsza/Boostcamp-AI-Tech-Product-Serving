import numpy as np
import torch
import torch.nn as nn


class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class FeaturesLinear(nn.Module):
    """
    Linear combination
    """
    def __init__(self, field_dims: np.ndarray, output_dim: int=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)

    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FactorizationMachine(nn.Module):
    def __init__(self, reduce_sum:bool=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x: torch.Tensor):
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class FactorizationMachineModel(nn.Module):
    def __init__(self, data, embed_dim):
        super().__init__()
        self.field_dims = data['field_dims']
        self.embedding = FeaturesEmbedding(self.field_dims, embed_dim)
        self.linear = FeaturesLinear(self.field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x: torch.Tensor):
        x = self.linear(x) + self.fm(self.embedding(x))
        return x.squeeze(1)
