import torch.nn as nn
from social_entity import SocialEntity


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

        for s_entity in self.population:
            # verify they are all people
            if not isinstance(s_entity, SocialEntity):
                raise TypeError("The argument \'population\' must only consist of type: SocialEntity")

            # set the loss functions
            s_entity.loss_func = self.add_loss
            s_entity.loss_iterator = self.iterate_loss_count
            self.models.append(s_entity)

    def add_loss(self, prediction, target, scale=1) -> None:
        self.loss += self.criterion(prediction, target) * scale

    def iterate_loss_count(self) -> None:
        self.loss_counter += 1

    def step(self, concept):
        self.train()

        for s_entity in self.population:
            s_entity.set_concept(concept=concept)

        for s_entity in self.population:
            if self.comm_type == 'rand':
                s_entity.receive_rand_choice()
            elif self.comm_type == 'weighted':
                s_entity.receive_weighted()

            if self.self_talk:
                # have the social_entity talk to themself, it should match
                s_entity.self_talk()

        loss = self.loss / self.loss_counter
        self.loss = 0
        self.loss_counter = 0
        return loss
