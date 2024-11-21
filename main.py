import torch
import torch.nn as nn
import torch.optim as optim
from population_grid import PopulationGrid
from train_population import train_population
from load_mnist import load_mnist_data
from display_population import display_population
from torch.utils.data import DataLoader
from synthetic_dataset import SyntheticDataset
import math
from assess_neighbors import average_hv_difference, average_diag_difference


def main():

    ROWS = 5
    COLS = 5
    EPOCHS = 10

    TEST_DATASET_SIZE = 100

    CONCEPT_SIZE = 28*28
    HIDDEN_SIZE = 2**(int(math.log2(CONCEPT_SIZE**(1/2))) + 2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #train_dataset, test_dataset = load_mnist_data()


    train_dataset = SyntheticDataset(
        size=5_000,
        num_dimensions=CONCEPT_SIZE,
        num_clusters=6,
        std_dev=0.15)

    test_dataset = SyntheticDataset(
        size=TEST_DATASET_SIZE,
        num_dimensions=CONCEPT_SIZE,
        num_clusters=6,
        std_dev=0.15)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    criterion = nn.MSELoss()

    population_grid = PopulationGrid(
        rows=ROWS,
        cols=COLS,
        concept_size=CONCEPT_SIZE,
        hidden_size=HIDDEN_SIZE,
        language_size=32,
        criterion=criterion,
        connection_type='neighbors adj',
        comm_type='weighted',
        self_talk=False).to(device)

    optimizer = optim.Adam(population_grid.parameters(), lr=0.001)

    train_population(population_grid, train_loader, optimizer, epochs=EPOCHS)

    display_population(population=population_grid, test_dataset=test_dataset, batch_size=TEST_DATASET_SIZE)

    grid_data = population_grid.get_output_grid(test_dataset=test_dataset, batch_size=TEST_DATASET_SIZE)

    direct_diff = 0
    diagonal_diff = 0

    for i in range(ROWS):
        for j in range(COLS):
            direct_diff += average_hv_difference(grid_data, ROWS, COLS, i, j)
            diagonal_diff += average_diag_difference(grid_data, ROWS, COLS, i, j)

    print("h or v: ", direct_diff)
    print("d:      ", diagonal_diff)


if __name__ == "__main__":
    main()
