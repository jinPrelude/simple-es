import multiprocessing as mp
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Agent(nn.Module):
    def __init__(
        self, obs_space, action_space, D_hidden=[80, 80],
    ):
        super(Agent, self).__init__()
        self.layers = []
        in_size = obs_space[0]
        D_hidden.append(action_space[0])
        for i, out_size in enumerate(D_hidden):
            tmp_layer = nn.Linear(in_size, out_size)
            with torch.no_grad():
                tmp_layer.weight = nn.init.normal_(tmp_layer.weight)
            in_size = out_size
            self.layers.append(tmp_layer)

    def forward(self, x):
        for hidden_layer in self.layers:
            x = hidden_layer(x)
        return x


class CNNAgent(nn.Module):
    def __init__(
        self,
        input_channel_size: int,
        channel_size: list,
        obs_space,
        action_space,
        D_hidden=[80, 80],
    ):
        super(CNNAgent, self).__init__()
        self.layers = []

        tmp_input_channel = input_channel_size

        self.cnn = nn.Sequential()
        for i, cnnlayer in enumerate(channel_size):
            tmp_layer = nn.Conv2d(
                tmp_input_channel, cnnlayer[0], cnnlayer[1], cnnlayer[2]
            )
            with torch.no_grad():
                tmp_layer.weight = nn.init.normal_(tmp_layer.weight)
            tmp_input_channel = cnnlayer[0]
            self.cnn.add_module("cnn_{}".format(i), tmp_layer)

        # calculate fc input size
        with torch.no_grad():
            test = torch.zeros(tuple(obs_space))
            test = torch.transpose(test, -1, 0).unsqueeze(0)
            test_output = self.cnn(test).detach().view(-1)
        in_size = test_output.shape[0]
        D_hidden.append(action_space[0])

        for j, out_size in enumerate(D_hidden):
            tmp_layer = nn.Linear(in_size, out_size)
            with torch.no_grad():
                tmp_layer.weight = nn.init.normal_(tmp_layer.weight)
            in_size = out_size
            self.__setattr__("hidden_fc{}".format(j), tmp_layer)
            self.layers.append(tmp_layer)

    def forward(self, x):
        x = torch.transpose(x, -1, 0).unsqueeze(0)
        x = self.cnn(x)
        x = torch.flatten(x)
        for hidden_layer in self.layers:
            x = hidden_layer(x)
        x = F.sigmoid(x)
        x[0] = (x[0] * 2) - 1.0
        return x
