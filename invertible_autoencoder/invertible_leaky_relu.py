import torch
from .invertible_layer import InvertibleLayer

class InvertibleLeakyReLU(InvertibleLayer):
    def __init__(self, negative_slope=0.01, *args, **kwargs):
        super(InvertibleLeakyReLU, self).__init__(*args, **kwargs)
        self.negative_slope = negative_slope

    def forward(self, x):
        return torch.where(x >= 0, x, self.negative_slope * x)

    def inverse(self, y):
        return torch.where(y >= 0, y, y / self.negative_slope)
