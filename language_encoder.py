import torch
import torch.nn as nn

class LanguageEncoder(nn.Module):

    def __init__(self, concept_size, hidden_size, message_size):
        super(LanguageEncoder, self).__init__()
        self.fc1 = nn.Linear(concept_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, message_size)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        x = torch.nn.functional.leaky_relu(self.fc2(x))
        x = torch.nn.functional.leaky_relu(self.fc3(x))
        return x
