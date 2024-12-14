from social_entity import SocialEntity
from .encoder_decoder import LanguageEncoder
from .encoder_decoder import LanguageDecoder


class StandardSocialEntity(SocialEntity):

    def __init__(self, perception_size, concept_size, hidden_size, message_size):
        super(StandardSocialEntity, self).__init__(
            perception_size=perception_size,
            concept_size=concept_size,
            hidden_size=hidden_size,
            message_size=message_size)

        self.language_encoder = LanguageEncoder(
            concept_size=concept_size,
            hidden_size=hidden_size,
            message_size=message_size)

        self.language_decoder = LanguageDecoder(
            message_size=message_size,
            hidden_size=hidden_size,
            concept_size=concept_size)

    def encode(self, x):
        return self.language_encoder(x)

    def decode(self, x):
        return self.language_decoder(x)
