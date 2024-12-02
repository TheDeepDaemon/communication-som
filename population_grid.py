import torch
import numpy as np
from person import StandardPerson, InvertiblePerson
from population_graph import PopulationGraph
from torch.utils.data import DataLoader


class PopulationGrid(PopulationGraph):

    def __init__(
            self,
            rows: int,
            cols: int,
            concept_size: int,
            hidden_size: int,
            message_size: int,
            criterion,
            connection_type: str,
            comm_type: str,
            self_talk: bool=True,
            person_type: str='standard'
    ) -> None:

        self.rows = rows
        self.cols = cols

        self.concept_size = concept_size
        self.message_size = message_size

        population = self.init_population(
            concept_size=concept_size,
            hidden_size=hidden_size,
            message_size=message_size,
            connection_type=connection_type,
            person_type=person_type)

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
            message_size: int,
            connection_type: str,
            person_type: str):

        population = []

        self.population_grid = []

        # actually create the grid with population members
        for i in range(self.rows):
            population_row = []
            for j in range(self.cols):

                # init the person object
                if person_type == 'invertible':
                    # invertible person
                    new_person = InvertiblePerson(
                        concept_size,
                        hidden_size,
                        message_size)
                else:
                    # default case
                    new_person = StandardPerson(
                        concept_size,
                        hidden_size,
                        message_size)

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
        elif connection_type == 'neighbors adj':
            for i in range(self.rows):
                for j in range(self.cols):

                    neighbors = []
                    for k in range(-1, 2):
                        for l in range(-1, 2):
                            if ((k != 0) or (l != 0)) and ((k == 0) or (l == 0)):
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

    def get_output_grid(self, test_dataset, batch_size: int):

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        dataset_size = len(test_dataset)

        output_collection = torch.zeros((dataset_size, self.rows, self.cols, self.message_size))

        self.eval()

        # iterate through all test data entries
        for batch_idx, inputs in enumerate(test_loader):

            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, dataset_size)

            concept = torch.flatten(inputs, start_dim=1)

            # iterate through the grid
            for i in range(self.rows):
                for j in range(self.cols):
                    person = self.population_grid[i][j]
                    person.set_concept(concept=concept)
                    encoded = person.encode_concept_to_language()
                    output_collection[start_idx:end_idx, i, j] = encoded.clone().detach()

        output_collection = output_collection.numpy()
        data_grid = output_collection.transpose((1, 2, 0, 3))
        data_grid = data_grid.reshape((*data_grid.shape[:2], -1))

        return data_grid

    def get_corners_error(self, test_dataset, batch_size):

        self.train()

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        for batch_idx, inputs in enumerate(test_loader):

            concept = torch.flatten(inputs, start_dim=1)

            corner_top_left = self.population_grid[0][0]
            corner_top_right = self.population_grid[0][-1]
            corner_bottom_left = self.population_grid[-1][0]
            corner_bottom_right = self.population_grid[-1][-1]

            corners = [corner_top_left, corner_top_right, corner_bottom_left, corner_bottom_right]

            # pass each concept through the others
            for i, member_i in enumerate(corners):
                for j, member_j in enumerate(corners):
                    if i != j:
                        # pass a message from one person to the other
                        member_i.set_concept(concept=concept)
                        member_j.receive_from(member_i)

        loss = self.loss / self.loss_counter
        self.loss = 0
        return float(loss)
