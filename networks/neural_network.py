import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

    def init_weights(self, mu=0, sigma=1):
        for param in self.parameters():
            nn.init.normal_(param, mu, sigma)

class IndirectEncoding(BaseNetwork):
    def __init__(self, num_state=8, num_action=4, discrete_action=True):
        super(IndirectEncoding, self).__init__()
        self.num_action = num_action
        self.num_state = num_state
        self.cppn1 = nn.Linear(4, 64)
        self.cppn2 = nn.Linear(64, 1)
        self.fc1 = nn.Linear(num_state, 32, bias=False)
        self.fc2 = nn.Linear(32, num_action, bias=False)
        self.discrete_action = discrete_action

    def forward(self, x):
        x = x.unsqueeze(0)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        if self.discrete_action:
            x = F.softmax(x.squeeze(), dim=0)
            x = torch.argmax(x)
        else:
            x = torch.tanh(x.squeeze())
        x = x.detach().cpu().numpy()
        return x

    def cppn_forward(self, x):
        x = torch.tanh(self.cppn1(x))
        x = torch.tanh(self.cppn2(x))*3.
        x = x.detach()
        return x

    def reset(self):
        l1 = np.array([[i/(self.num_state*2), 0.25] for i in range(1, self.num_state+1)])
        l2 = np.array([[i/(32*2), 0.5] for i in range(1, 33)])
        l3 = np.array([[i/(self.num_action*2), 0.75] for i in range(1, self.num_action+1)])
        fc1 = torch.zeros(32, self.num_state).float()
        fc2 = torch.zeros(self.num_action, 32).float()
        for idx_i, i in enumerate(l2):
            for idx_j, j in enumerate(l1):
                cppn_input = torch.tensor(np.concatenate((j, i))).float()
                fc1[idx_i][idx_j] = self.cppn_forward(cppn_input)
        
        for idx_i, i in enumerate(l3):
            for idx_j, j in enumerate(l2):
                cppn_input = torch.tensor(np.concatenate((j, i))).float()
                fc2[idx_i][idx_j] = self.cppn_forward(cppn_input)
        self.fc1.weight.data = fc1.clone().detach()
        self.fc2.weight.data = fc2.clone().detach()

    def init_weights(self, mu=0, sigma=1):
        pass
        # for param in self.parameters():
        #     nn.init.normal_(param, mu, sigma)

    def parameters(self, cppn=True):
        p_lst = []
        for p in super().parameters():
            p_lst.append(p)
        if cppn:
            for p in p_lst[:4]:
                yield p
        else:
            for p in p_lst[4:]:
                yield p