import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()

        hidden_size = 100

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, encoding_dim)
        )

        # Decoder: Encoded -> Hidden -> Output
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def generate_nd_data(size, num_dimensions, num_clusters):
    centers = np.random.randint(0, 2, (num_clusters, num_dimensions)).astype(dtype=float)
    data = np.random.standard_normal((size, num_dimensions)) * 0.15

    for i in range(size):
        index = np.random.choice(np.arange(num_clusters))
        data[i] += centers[index]

    data_tensor = torch.tensor(data, dtype=torch.float32)

    return data_tensor


size = 1000
num_dimensions = 10
num_clusters = 5

data_tensor = generate_nd_data(size, num_dimensions, num_clusters)

batch_size = 64
dataset = TensorDataset(data_tensor, data_tensor)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

input_dim = num_dimensions
encoding_dim = 3
autoencoder = Autoencoder(input_dim=input_dim, encoding_dim=encoding_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)



num_epochs = 100
for epoch in range(num_epochs):
    for data_batch, _ in data_loader:
        outputs = autoencoder(data_batch)
        loss = criterion(outputs, data_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# Test the autoencoder on some data (just showing an example with the first few points)
with torch.no_grad():
    example_data = data_tensor[:5]
    reconstructed_data = autoencoder(example_data)

    print("\nOriginal Data: ")
    print(example_data.numpy())

    print("\nReconstructed Data: ")
    print(reconstructed_data.numpy())

    print(f"loss: {torch.mean(torch.square(example_data - reconstructed_data))}")
