import torch
import torch.nn as nn
from language_encoder import LanguageEncoder
from language_decoder import LanguageDecoder
import random

class Person(nn.Module):

    def __init__(self, concept_size, hidden_size, language_size):
        super(Person, self).__init__()

        self.perceived_concept = torch.zeros(concept_size)

        self.language_encoder = LanguageEncoder(
            concept_size=concept_size,
            hidden_size=hidden_size,
            language_size=language_size)

        self.language_decoder = LanguageDecoder(
            language_size=language_size,
            hidden_size=hidden_size,
            concept_size=concept_size)

        self.neighbors = []

    def set_concept(self, concept):
        self.perceived_concept = concept

    def encode_concept_to_language(self):
        return self.language_encoder(self.perceived_concept)

    def decode_concept_from_language(self, language_data):
        return self.language_decoder(language_data)

    def add_neighbor(self, person):
        self.neighbors.append(person)

    def receive_from_neighbor(self):
        neighbor = random.choice(self.neighbors)
        linguistic_info = neighbor.encode_concept_to_language()
        communicated_concept = self.decode_concept_from_language(linguistic_info)
        return communicated_concept

    def self_talk(self):
        linguistic_info = self.encode_concept_to_language()
        communicated_concept = self.decode_concept_from_language(linguistic_info)
        return communicated_concept
