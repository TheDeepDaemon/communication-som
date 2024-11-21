import torch
import torch.nn as nn
import torch.optim as optim
from population_grid import PopulationGrid
from train_population import train_population
from load_mnist import load_mnist_data
from synthetic_dataset import create_synthetic_data
from display_population import display_population
from assess_neighbors import average_hv_difference, average_diag_difference


def main():

    ROWS = 5
    COLS = 5
    EPOCHS = 10

    TEST_BATCH_SIZE = 100

    USE_SYNTHETIC_DATA = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if USE_SYNTHETIC_DATA:

        CONCEPT_SIZE = 128
        HIDDEN_SIZE = 32

        train_dataset, test_dataset = create_synthetic_data(
            train_size=5_000,
            test_size=TEST_BATCH_SIZE,
            num_dimensions=CONCEPT_SIZE,
            num_clusters=10,
            std_dev=0.15)

    else:

        CONCEPT_SIZE = 28*28
        HIDDEN_SIZE = 64

        train_dataset, test_dataset = load_mnist_data()

    criterion = nn.MSELoss()

    population_grid = PopulationGrid(
        rows=ROWS,
        cols=COLS,
        concept_size=CONCEPT_SIZE,
        hidden_size=HIDDEN_SIZE,
        message_size=6,
        criterion=criterion,
        connection_type='neighbors adj',
        comm_type='weighted',
        self_talk=False).to(device)

    optimizer = optim.Adam(population_grid.parameters(), lr=0.001)

    train_population(population_grid, train_dataset=train_dataset, optimizer=optimizer, epochs=EPOCHS, batch_size=64)

    display_population(population=population_grid, test_dataset=test_dataset, batch_size=TEST_BATCH_SIZE)

    grid_data = population_grid.get_output_grid(test_dataset=test_dataset, batch_size=TEST_BATCH_SIZE)

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
