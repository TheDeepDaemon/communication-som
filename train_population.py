import torch
from population_graph import PopulationGraph


def train_population(model: PopulationGraph, train_loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs in train_loader:

            optimizer.zero_grad()

            loss = model.step(torch.flatten(inputs, start_dim=1))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.8f}')
