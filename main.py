import torch
import torch.nn as nn
import torch.optim as optim
from population_grid import PopulationGrid
from train_population import train_population
from load_mnist import load_mnist_data
from display_population import display_population
from torch.utils.data import DataLoader
from synthetic_dataset import SyntheticDataset


def main():

    ROWS = 5
    COLS = 5
    EPOCHS = 10

    CONCEPT_SIZE = 100
    HIDDEN_SIZE = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #train_dataset, test_dataset = load_mnist_data()

    train_dataset = SyntheticDataset(size=5_000, num_dimensions=CONCEPT_SIZE, num_clusters=6, std_dev=0.15)
    test_dataset = SyntheticDataset(size=10, num_dimensions=CONCEPT_SIZE, num_clusters=6, std_dev=0.15)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    criterion = nn.MSELoss()

    population_grid = PopulationGrid(
        rows=ROWS,
        cols=COLS,
        concept_size=CONCEPT_SIZE,
        hidden_size=HIDDEN_SIZE,
        language_size=10,
        criterion=criterion,
        connection_type='neighbors even',
        comm_type='weighted',
        self_talk=True).to(device)

    optimizer = optim.Adam(population_grid.parameters(), lr=0.001)

    train_population(population_grid, train_loader, optimizer, epochs=EPOCHS)

    display_population(population=population_grid, test_dataset=test_dataset, batch_size=1000)


if __name__ == "__main__":
    main()
