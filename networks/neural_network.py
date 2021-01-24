import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNModel(nn.Module):
    def __init__(self, in_channels=4, num_actions=18):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)


class EatAppleModel(nn.Module):
    def __init__(self, in_channels=1, num_action=4):
        super().__init__()
        self.num_action = num_action
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(16, 32)
        self.gru = nn.GRU(32, 32 + num_action)
        self.fc2 = nn.Linear(32 + num_action, num_action)
        self.last_action_onehot = torch.zeros((1, 1, self.num_action))

    def forward(self, x, h):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), 1, -1)
        x = F.relu(self.fc1(x))

        x, h = self.gru(x, h)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x.squeeze(), dim=0)

        action = torch.argmax(x)
        self.last_action_onehot = torch.zeros((1, 1, self.num_action))
        self.last_action_onehot[0, 0, action] = 1
        return x, h

    def init_hidden(self):
        return torch.zeros([1, 1, 32 + self.num_action], dtype=torch.float)

    def init_weights(self, mu=0, sigma=1):
        for param in self.parameters():
            nn.init.normal_(param, mu, sigma)


class GymEnvModel(nn.Module):
    def __init__(self, num_state=8, num_action=4):
        super().__init__()
        self.num_action = num_action
        self.fc1 = nn.Linear(num_state, 32)
        self.gru = nn.GRU(32, 32 + num_action)
        self.fc2 = nn.Linear(32 + num_action, num_action)
        self.last_action_onehot = torch.zeros((1, 1, self.num_action))

    def forward(self, x, h):
        x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x, h = self.gru(x, h)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x.squeeze(), dim=0)

        action = torch.argmax(x)
        self.last_action_onehot = torch.zeros((1, 1, self.num_action))
        self.last_action_onehot[0, 0, action] = 1
        return x, h

    def init_hidden(self):
        return torch.zeros([1, 1, 32 + self.num_action], dtype=torch.float)

    def init_weights(self, mu=0, sigma=1):
        for param in self.parameters():
            nn.init.normal_(param, mu, sigma)
