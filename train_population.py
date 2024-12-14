import torch
from torch.utils.data import DataLoader
from population_graph import PopulationGraph


def train_population(model: PopulationGraph, train_dataset, optimizer, epochs, batch_size):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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


def eval_population(model: PopulationGraph, test_dataset, optimizer, epochs, batch_size):

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    running_loss = 0.0
    for inputs in test_loader:

        optimizer.zero_grad()

        loss = model.step(torch.flatten(inputs, start_dim=1))

        running_loss += loss.item()
    return running_loss / len(test_loader)
