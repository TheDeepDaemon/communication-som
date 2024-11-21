from person import Person
from population_graph import PopulationGraph
import numpy as np


class PopulationGrid(PopulationGraph):

    def __init__(
            self,
            rows: int,
            cols: int,
            concept_size: int,
            hidden_size: int,
            language_size: int,
            criterion,
            connection_type: str,
            comm_type: str,
            self_talk: bool=True
    ) -> None:

        self.rows = rows
        self.cols = cols

        self.concept_size = concept_size
        self.language_size = language_size

        population = self.init_population(
            concept_size,
            hidden_size,
            language_size,
            connection_type)

        super(PopulationGrid, self).__init__(
            population=population,
            criterion=criterion,
            comm_type=comm_type,
            self_talk=self_talk)

    def in_grid_bounds(self, row_index: int, col_index: int) -> bool:
        return (
                (row_index >= 0) and
                (col_index >= 0) and
                (row_index < self.rows) and
                (col_index < self.cols))

    def init_population(
            self,
            concept_size: int,
            hidden_size: int,
            language_size: int,
            connection_type: str):

        population = []

        self.population_grid = []

        # actually create the grid with population members
        for i in range(self.rows):
            population_row = []
            for j in range(self.cols):

                # init the person object
                new_person = Person(
                    concept_size,
                    hidden_size,
                    language_size)

                # add to this row
                population_row.append(new_person)

                # add this to all models (so it tracks the parameters)
                population.append(new_person)

            # add the whole row
            self.population_grid.append(population_row)

        # connect to other people based on the communication type
        if connection_type == 'neighbors even':
            for i in range(self.rows):
                for j in range(self.cols):

                    neighbors = []
                    for k in range(-1, 2):
                        for l in range(-1, 2):
                            if (k != 0) or (l != 0):
                                row_index = i + k
                                col_index = j + l
                                if self.in_grid_bounds(row_index, col_index):
                                    neighbors.append(self.population_grid[row_index][col_index])

                    self.population_grid[i][j].set_neighbors(
                        neighbors=neighbors,
                        weightings=np.ones(len(neighbors), dtype=float))
        elif connection_type == 'all dist': # connect all nodes, make them distance weighted
            for i in range(self.rows):
                for j in range(self.cols):
                    this_pos = np.array([j, i], dtype=float)

                    neighbors = []
                    weightings = []

                    for k in range(self.rows):
                        for l in range(self.cols):

                            if (i != k) or (j != l):

                                neighbors.append(self.population_grid[k][l])

                                other_pos = np.array([l, k], dtype=float)
                                dist = np.linalg.norm(this_pos - other_pos)
                                weighting = np.exp(-dist)
                                weightings.append(weighting)

                    self.population_grid[i][j].set_neighbors(
                        neighbors=neighbors,
                        weightings=np.array(weightings, dtype=float))

        return population

