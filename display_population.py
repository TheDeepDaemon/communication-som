import numpy as np
from population_grid import PopulationGrid
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def display_population(population: PopulationGrid, test_dataset, batch_size: int):

    data_grid = population.get_output_grid(test_dataset=test_dataset, batch_size=batch_size)

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
