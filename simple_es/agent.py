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
            self.__setattr__("hidden_fc{}".format(i), tmp_layer)
            self.layers.append(tmp_layer)

    def forward(self, x):
        for hidden_layer in self.layers:
            x = hidden_layer(x)
        return x

    def get_np_params(self):
        np_layer = []
        for layer in self.layers:
            np_layer.append(layer.weight.data.numpy())
        return np_layer

    def initialize_params(self, mu, std):
        for i in range(len(self.layers)):
            with torch.no_grad():
                if isinstance(mu, list):
                    weight = np.random.normal(mu[i], std)
                else:
                    weight = np.random.normal(
                        mu, np.ones(self.layers[i].weight.shape) * std
                    )
                weight = torch.Tensor(weight).float()
                self.layers[i].weight.data = weight


class HebbianAgent(nn.Module):
    def __init__(
        self, obs_space, action_space, D_hidden=[80, 80],
    ):
        super(HebbianAgent, self).__init__()
        self.layers = []
        self.hebbian_coefficient = torch.ones(5)  # a, b, c, d, lr
        self.hebbian_coefficient = nn.init.uniform_(self.hebbian_coefficient, 0.0, 1.0)
        in_size = obs_space[0]
        D_hidden.append(action_space[0])
        for i, out_size in enumerate(D_hidden):
            tmp_layer = nn.Linear(in_size, out_size)
            with torch.no_grad():
                tmp_layer.weight = nn.init.uniform_(tmp_layer.weight, -0.1, 0.1)
            in_size = out_size
            self.__setattr__("hidden_fc{}".format(i), tmp_layer)
            self.layers.append(tmp_layer)

    def update_params(self, x, hidden_layer):
        with torch.no_grad():
            result = hidden_layer(x)
            result = F.tanh(result)
            tmp = torch.matmul(
                result.unsqueeze(-1), (self.hebbian_coefficient[0] * x).unsqueeze(0)
            )
            tmp = torch.add(
                tmp,
                (self.hebbian_coefficient[1] * x)
                .unsqueeze(0)
                .repeat(result.shape[0], 1),
            )
            tmp = torch.add(
                tmp,
                (self.hebbian_coefficient[2] * result)
                .unsqueeze(-1)
                .repeat(1, x.shape[0]),
            )
            tmp += self.hebbian_coefficient[3]
            tmp *= self.hebbian_coefficient[4]
            return result, tmp

    def forward(self, x):
        for i in range(len(self.layers)):
            x, new_layer_weight = self.update_params(x, self.layers[i])
            self.layers[i].weight.data = deepcopy(new_layer_weight)
        x = F.tanh(x)  # tanh for bipedalwalker
        return x

    def get_np_params(self):
        return self.hebbian_coefficient.detach().numpy()

    def initialize_params(self, mu, std):
        with torch.no_grad():
            tmp_hebbian_coefficient = np.random.uniform(
                0.0, 0.1, self.hebbian_coefficient.shape
            )


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
