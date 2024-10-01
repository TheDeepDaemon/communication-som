import torch.nn as nn
from person import Person


class PopulationGrid(nn.Module):

    def __init__(self, rows, cols, concept_size, hidden_size, language_size):
        super(PopulationGrid, self).__init__()
        self.rows = rows
        self.cols = cols
        self.concept_size = concept_size

        self.population = []
        self.models = nn.ModuleList()

        for _ in range(rows):
            population_row = []
            for _ in range(cols):
                new_person = Person(concept_size, hidden_size, language_size)
                population_row.append(new_person)
                self.models.append(new_person)
            self.population.append(population_row)

        for i in range(rows):
            for j in range(cols):

                if i < rows - 1:
                    self.population[i][j].add_neighbor(self.population[i+1][j])

                if i > 0:
                    self.population[i][j].add_neighbor(self.population[i-1][j])

                if j < cols - 1:
                    self.population[i][j].add_neighbor(self.population[i][j+1])

                if j > 0:
                    self.population[i][j].add_neighbor(self.population[i][j-1])

    def step(self, concept, criterion):
        self.train()

        for i in range(self.rows):
            for j in range(self.cols):
                self.population[i][j].set_concept(concept=concept)

        loss = 0

        for i in range(self.rows):
            for j in range(self.cols):
                person = self.population[i][j]

                # pick a neighbor to describe the concept
                communicated_concept = person.receive_from_neighbor()
                loss += criterion(communicated_concept, person.perceived_concept)

                # have the person talk to themself, it should match
                self_communicated_concept = person.self_talk()
                loss += criterion(self_communicated_concept, person.perceived_concept)

        return loss
