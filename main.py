import torch
import torch.nn as nn
import torch.optim as optim
from population_grid import PopulationGrid
from train_population import train_population, eval_population
from load_mnist import load_mnist_data
from synthetic_dataset import create_synthetic_data
from display_population import display_population
from assess_neighbors import average_hv_difference, average_diag_difference
import os


def main():

    RESULTS_FNAME = "results.csv"

    ROWS = 5
    COLS = 5
    EPOCHS = 20

    TEST_BATCH_SIZE = 100

    USE_SYNTHETIC_DATA = False

    SOCIAL_ENTITY_TYPE = 'standard'

    USE_SELF_TALK = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if USE_SYNTHETIC_DATA:

        PERCEPTION_SIZE = 128
        CONCEPT_SIZE = 48
        HIDDEN_SIZE_CM = 32 # CM: concept-message
        MESSAGE_SIZE = 8

        train_dataset, test_dataset = create_synthetic_data(
            train_size=5_000,
            test_size=TEST_BATCH_SIZE,
            num_dimensions=PERCEPTION_SIZE,
            num_clusters=4,
            std_dev=0.15)

    else:

        PERCEPTION_SIZE = 28*28
        CONCEPT_SIZE = 128
        HIDDEN_SIZE_CM = 64 # CM: concept-message
        MESSAGE_SIZE = 32

        train_dataset, test_dataset = load_mnist_data()

    criterion = nn.MSELoss()

    population_grid = PopulationGrid(
        rows=ROWS,
        cols=COLS,
        perception_size=PERCEPTION_SIZE,
        concept_size=CONCEPT_SIZE,
        hidden_size=HIDDEN_SIZE_CM,
        message_size=MESSAGE_SIZE,
        criterion=criterion,
        connection_type='neighbors adj',
        comm_type='rand',
        self_talk=USE_SELF_TALK,
        social_entity_type=SOCIAL_ENTITY_TYPE).to(device)

    optimizer = optim.Adam(population_grid.parameters(), lr=0.001)

    train_population(population_grid, train_dataset=train_dataset, optimizer=optimizer, epochs=EPOCHS, batch_size=64)

    display_population(population=population_grid, test_dataset=test_dataset, batch_size=TEST_BATCH_SIZE)

    grid_data = population_grid.get_output_grid(test_dataset=test_dataset, batch_size=TEST_BATCH_SIZE)

    direct_diff = 0
    diagonal_diff = 0
    count = 0

    for i in range(ROWS):
        for j in range(COLS):
            direct_diff += average_hv_difference(grid_data, ROWS, COLS, i, j)
            diagonal_diff += average_diag_difference(grid_data, ROWS, COLS, i, j)
            count += 1

    hv_dist = direct_diff / count
    diag_dist = diagonal_diff / count

    corners_loss = population_grid.get_corners_error(test_dataset, batch_size=TEST_BATCH_SIZE)

    baseline_loss = eval_population(
        population_grid, test_dataset=test_dataset, optimizer=optimizer, epochs=1, batch_size=TEST_BATCH_SIZE)

    if not os.path.exists(RESULTS_FNAME):
        with open("results.csv", 'w') as file:
            file.write("horizontal-vertical,diagonal,corners_loss,baseline_loss\n")

    with open("results.csv", 'a') as file:
        file.write(f"{hv_dist},{diag_dist},{corners_loss},{baseline_loss}\n")


if __name__ == "__main__":
    main()
