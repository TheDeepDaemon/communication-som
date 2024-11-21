import torch
import numpy as np
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    def __init__(self, size, num_dimensions, num_clusters, std_dev=0.15):
        self.size = size
        self.num_dimensions = num_dimensions
        self.num_clusters = num_clusters
        self.std_dev = std_dev
        self.centers = np.random.randint(0, 2, (num_clusters, num_dimensions)).astype(dtype=np.float32)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        if hasattr(idx, '__len__'):
            batch_size = len(idx)
        else:
            batch_size = 1

        data = np.random.standard_normal((batch_size, self.num_dimensions)) * self.std_dev

        for i in range(batch_size):
            index = np.random.choice(np.arange(self.num_clusters))
            data[i] += self.centers[index]

        return torch.tensor(data, dtype=torch.float32)
