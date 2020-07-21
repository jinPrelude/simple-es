from copy import deepcopy
import multiprocessing as mp
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Agent(nn.Module):
    def __init__(
        self, D_in=2, D_out=4, D_hidden=[80, 80],
    ):
        super(Agent, self).__init__()
        self.layers = []
        in_size = D_in
        D_hidden.append(D_out)
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
