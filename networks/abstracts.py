from abc import *

from torch import nn


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @abstractmethod
    def init_weights(self):
        pass

    @abstractmethod
    def reset(self):
        pass
