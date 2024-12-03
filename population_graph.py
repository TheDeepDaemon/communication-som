import torch.nn as nn
from person import Person


class PopulationGraph(nn.Module):

    def __init__(
            self,
            population: list,
            criterion,
            comm_type: str,
            self_talk: bool=True) -> None:
        super(PopulationGraph, self).__init__()

        self.population = population
        self.models = nn.ModuleList()
        self.loss = 0
        self.loss_counter = 0
        self.criterion = criterion
        self.self_talk = self_talk
        self.comm_type = comm_type

        for person in self.population:
            # verify they are all people
            if not isinstance(person, Person):
                raise TypeError("The argument \'population\' must only consist of type: Person")

            # set the loss functions
            person.loss_func = self.add_loss
            person.loss_iterator = self.iterate_loss_count
            self.models.append(person)

    def add_loss(self, prediction, target, scale=1) -> None:
        self.loss += self.criterion(prediction, target) * scale

    def iterate_loss_count(self) -> None:
        self.loss_counter += 1

    def step(self, concept):
        self.train()

        for person in self.population:
            person.set_concept(concept=concept)

        for person in self.population:
            if self.comm_type == 'rand':
                person.receive_rand_choice()
            elif self.comm_type == 'weighted':
                person.receive_weighted()

            if self.self_talk:
                # have the person talk to themself, it should match
                person.self_talk()

        loss = self.loss / self.loss_counter
        self.loss = 0
        return loss
