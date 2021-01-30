import torch
import torch.nn as nn
import torch.nn.functional as F

from .abstracts import BaseNetwork


class EatAppleModel(BaseNetwork):
    def __init__(self, in_channels=1, num_action=4, discrete_action=True):
        super(EatAppleModel, self).__init__()
        self.num_action = num_action
        self.discrete_action = discrete_action
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(16, 32)
        self.gru = nn.GRU(32, 3)
        self.fc2 = nn.Linear(32, num_action)

        self.h = torch.zeros([1, 1, 32], dtype=torch.float)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), 1, -1)
        x = F.relu(self.fc1(x))

        x, self.h = self.gru(x, self.h)
        x = F.relu(x)
        x = self.fc2(x)
        if self.discrete_action:
            x = F.softmax(x.squeeze(), dim=0)
            x = torch.argmax(x)
        else:
            x = torch.tanh(x.squeeze())
        x = x.detach().cpu().numpy()
        return x

    def reset(self):
        self.h = torch.zeros([1, 1, 32], dtype=torch.float)

    def init_weights(self, mu=0, sigma=1):
        for param in self.parameters():
            nn.init.normal_(param, mu, sigma)


class GymEnvModel(BaseNetwork):
    def __init__(self, num_state=8, num_action=4, discrete_action=True):
        super(GymEnvModel, self).__init__()
        self.num_action = num_action
        self.fc1 = nn.Linear(num_state, 32)
        self.gru = nn.GRU(32, 32)
        self.fc2 = nn.Linear(32, num_action)
        self.discrete_action = discrete_action

        self.h = torch.zeros([1, 1, 32], dtype=torch.float)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x, self.h = self.gru(x, self.h)
        x = F.relu(x)
        x = self.fc2(x)
        if self.discrete_action:
            x = F.softmax(x.squeeze(), dim=0)
            x = torch.argmax(x)
        else:
            x = torch.tanh(x.squeeze())
        x = x.detach().cpu().numpy()

        return x

    def reset(self):
        self.h = torch.zeros([1, 1, 32], dtype=torch.float)

    def init_weights(self, mu=0, sigma=1):
        for param in self.parameters():
            nn.init.normal_(param, mu, sigma)
