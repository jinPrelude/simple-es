import torch
import torch.nn as nn
import torch.nn.functional as F

from .abstracts import BaseNetwork


class GymEnvModel(BaseNetwork):
    def __init__(self, num_state=8, num_action=4, discrete_action=True, gru=True):
        super(GymEnvModel, self).__init__()
        self.num_action = num_action
        self.fc1 = nn.Linear(num_state, 32)
        self.use_gru = gru
        if self.use_gru:
            self.gru = nn.GRU(32, 32)
            self.h = torch.zeros([1, 1, 32], dtype=torch.float)
        self.fc2 = nn.Linear(32, num_action)
        self.discrete_action = discrete_action

    def forward(self, x):
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            x = x.unsqueeze(0)
            x = torch.tanh(self.fc1(x))
            if self.use_gru:
                x, self.h = self.gru(x, self.h)
                x = torch.tanh(x)
            x = self.fc2(x)
            if self.discrete_action:
                x = F.softmax(x.squeeze(), dim=0)
                x = torch.argmax(x)
            else:
                x = torch.tanh(x.squeeze())
            x = x.detach().cpu().numpy()

            return x

    def reset(self):
        if self.use_gru:
            self.h = torch.zeros([1, 1, 32], dtype=torch.float)

    def zero_init(self):
        for param in self.parameters():
            param.data = torch.zeros(param.shape)

    def get_param_list(self):
        param_list = []
        for param in self.parameters():
            param_list.append(param.data.numpy())
        return param_list

    def apply_param(self, param_lst: list):
        count = 0
        for p in self.parameters():
            p.data = torch.tensor(param_lst[count]).float()
            count += 1


class AtariDQN(BaseNetwork):
    def __init__(self, num_channel=4, num_action=14):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(AtariDQN, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 32, kernel_size=8, stride=4)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, num_action)

    def forward(self, x):
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            x = x.float() / 255
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.fc4(x.view(x.size(0), -1)))
            x = self.head(x)
            x = torch.argmax(x)
            x = x.detach().cpu().numpy()
            return x

    def zero_init(self):
        for param in self.parameters():
            param.data = torch.zeros(param.shape)

    def reset(self):
        pass

    def get_param_list(self):
        param_list = []
        for param in self.parameters():
            param_list.append(param.data.numpy())
        return param_list

    def apply_param(self, param_lst: list):
        count = 0
        for p in self.parameters():
            p.data = torch.tensor(param_lst[count]).float()
            count += 1
