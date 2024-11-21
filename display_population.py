import numpy as np
import torch
from population_grid import PopulationGrid
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def display_population(population: PopulationGrid, test_dataset, batch_size):

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    output_collection = torch.zeros((len(test_dataset), population.rows, population.cols, population.language_size))

    population.eval()

    # iterate through all test data entries
    for batch_idx, inputs in enumerate(test_loader):

        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size

        concept = torch.flatten(inputs, start_dim=1)

        # iterate through the grid
        for i in range(population.rows):
            for j in range(population.cols):
                person = population.population_grid[i][j]
                person.set_concept(concept=concept)
                encoded = person.encode_concept_to_language()
                output_collection[start_idx:end_idx, i, j] = encoded.clone().detach()

    output_collection = output_collection.numpy()
    data_grid = output_collection.transpose((1, 2, 0, 3))
    data_grid = data_grid.reshape((*data_grid.shape[:2], -1))

    inner_dim = data_grid.shape[:2]
    data_mat = data_grid.reshape((-1, data_grid.shape[-1]))

    pca = PCA(n_components=3)
    transformed = pca.fit_transform(data_mat)
    transformed = transformed.reshape((*inner_dim, 3))
    transformed -= np.min(transformed)
    transformed /= np.max(transformed)

    plt.imshow(transformed)
    plt.axis('off')
    plt.show()
