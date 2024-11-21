import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        h_size = 2**3
        # Input layer to hidden layer with 2 nodes
        self.hidden = nn.Linear(1, h_size)
        # Hidden layer to output layer
        self.output = nn.Linear(h_size, 1)
        # Activation function
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x

model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 10_000
for epoch in range(epochs):

    x = (torch.rand((100, 1)) - 0.5) * 3.0

    y = x ** 2

    y_pred = model(x)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')


x = torch.linspace(-1, 1, 100).view(-1, 1)
y = x ** 2

y_pred = model(x).detach()
plt.scatter(x, y, label='True Data')
plt.plot(x, y_pred, color='red', label='Model Prediction')
plt.legend()
plt.show()