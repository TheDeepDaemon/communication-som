import torch.nn as nn
from .invertible_layer import InvertibleLayer
from typing import List

class InvertibleAutoencoder(nn.Module):

    def __init__(self, layers: List[InvertibleLayer], *args, **kwargs):
        super(InvertibleAutoencoder, self).__init__(*args, **kwargs)
        self.layers = nn.ModuleList()
        for layer in layers:
            self.layers.append(layer)

    def encode(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def decode(self, y):
        for layer in reversed(self.layers):
            y = layer.inverse(y)

        return y

    def forward(self, x):
        latent = self.encode(x)
        return self.decode(latent)
