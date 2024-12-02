import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class Person(nn.Module, ABC):

    def __init__(
            self, concept_size):
        super(Person, self).__init__()

        self.perceived_concept = torch.zeros(concept_size)

        self.neighbors = None
        self.weightings = None

        self.has_encoded = False
        self.encoded_message = None

    @abstractmethod
    def encode(self, x):
        pass

    @abstractmethod
    def decode(self, x):
        pass

    def set_concept(self, concept):
        self.perceived_concept = concept
        self.has_encoded = False

    def encode_concept_to_language(self):

        # encode if hasn't already
        if not self.has_encoded:
            # go through encoder model
            self.encoded_message = self.encode(self.perceived_concept)

            # update variable
            self.has_encoded = True

        # return the message
        return self.encoded_message

    def decode_concept_from_language(self, language_data):
        return self.decode(language_data)

    def set_neighbors(self, neighbors: list, weightings: np.ndarray):
        self.neighbors = neighbors

        # make sure the neighbor's weights add up to one
        self.weightings = weightings / np.sum(weightings)

    def receive_weighted(self):

        for neighbor, weight in zip(self.neighbors, self.weightings):

            # get the message
            message = neighbor.encode_concept_to_language()

            # convert it to a concept
            communicated_concept = self.decode_concept_from_language(message)

            # add to loss
            self.loss_func(communicated_concept, self.perceived_concept, weight)

        # iterate number of losses
        self.loss_iterator()

    def receive_rand_choice(self):
        speaker = np.random.choice(self.neighbors, p=self.weightings)
        self.receive_from(speaker)

    def receive_from(self, speaker):
        message = speaker.encode_concept_to_language()
        communicated_concept = self.decode_concept_from_language(message)
        self.loss_func(communicated_concept, self.perceived_concept)
        self.loss_iterator()

    def self_talk(self):
        self.receive_from(self)