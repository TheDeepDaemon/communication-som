from .invertible_layer import InvertibleLayer
import torch.nn as nn
import torch


class BiasLayer(InvertibleLayer):

    def __init__(self, size):
        super(BiasLayer, self).__init__()
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        return x + self.bias

    def inverse(self, y):
        return y - self.bias
