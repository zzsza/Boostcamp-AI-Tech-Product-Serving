import numpy as np
import torch
import torch.nn as nn


class FactorizationMachine(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.v = nn.Parameter(torch.rand(input_dim, latent_dim), requires_grad = True)
        self.linear = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        linear = self.linear(x)
        square_of_sum = torch.mm(x, self.v) ** 2
        sum_of_square = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        output = linear + (0.5 * pair_interactions)
        return output


class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class CNNBase(nn.Module):
    def __init__(self, ):
        super(CNNBase, self).__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x):
        x = self.cnn_layer(x)
        x = x.view(-1, 12 * 1 * 1)
        return x


class CNNFM(torch.nn.Module):
    def __init__(self, data, cnn_embed_dim, cnn_latent_dim):
        super().__init__()
        self.field_dims = np.array([len(data['user2idx']), len(data['isbn2idx'])], dtype=np.uint32)
        self.embedding = FeaturesEmbedding(self.field_dims, cnn_embed_dim)
        self.cnn = CNNBase()
        self.fm = FactorizationMachine(
            input_dim=(cnn_embed_dim * 2) + (12 * 1 * 1),
            latent_dim=cnn_latent_dim,
        )

    def forward(self, x):
        user_isbn_vector, img_vector = x[0], x[1]
        user_isbn_feature = self.embedding(user_isbn_vector)
        img_feature = self.cnn(img_vector)
        feature_vector = torch.cat(
            [user_isbn_feature.view(-1, user_isbn_feature.size(1) * user_isbn_feature.size(2)),img_feature],
            dim=1
        )
        output = self.fm(feature_vector)
        return output.squeeze(1)
