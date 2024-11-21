import torch.nn as nn
from abc import ABC, abstractmethod

class InvertibleLayer(nn.Module, ABC):

    def __init__(self, *args, **kwargs):
        super(InvertibleLayer, self).__init__(*args, **kwargs)

    @abstractmethod
    def inverse(self, y):
        pass
