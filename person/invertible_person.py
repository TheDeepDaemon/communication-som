from person import Person
from .invertible_autoencoder import InvertibleAutoencoder, PseudoInvertibleLayer, BiasLayer, InvertibleLeakyReLU


class InvertiblePerson(Person):

    def __init__(self, perception_size,  concept_size, hidden_size, message_size):
        super(InvertiblePerson, self).__init__(
            perception_size=perception_size,
            concept_size=concept_size,
            hidden_size=hidden_size,
            message_size=message_size)

        layers_list = []
        layers_list.append(PseudoInvertibleLayer(in_features=concept_size, out_features=hidden_size))
        layers_list.append(BiasLayer(hidden_size))
        layers_list.append(InvertibleLeakyReLU())
        layers_list.append(PseudoInvertibleLayer(in_features=hidden_size, out_features=message_size))
        layers_list.append(BiasLayer(message_size))

        self.autoencoder = InvertibleAutoencoder(layers=layers_list)

    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, x):
        return self.autoencoder.decode(x)
