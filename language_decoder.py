import torch
import torch.nn as nn

class LanguageDecoder(nn.Module):

    def __init__(self, message_size, hidden_size, concept_size):
        super(LanguageDecoder, self).__init__()
        self.fc1 = nn.Linear(message_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, concept_size)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        x = torch.nn.functional.leaky_relu(self.fc2(x))
        x = torch.nn.functional.leaky_relu(self.fc3(x))
        return x
