import torch
import torch.nn as nn
from .invertible_layer import InvertibleLayer

class PseudoInvertibleLayer(InvertibleLayer):
    """
    A layer that uses the Moore-Penrose pseudoinverse as it's inverse.
    """

    def __init__(self, in_features, out_features, *args, **kwargs):
        super(PseudoInvertibleLayer, self).__init__(*args, **kwargs)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        weight_pinv = torch.linalg.pinv(self.weight)
        return x @ weight_pinv

    def inverse(self, x):
        return x @ self.weight
