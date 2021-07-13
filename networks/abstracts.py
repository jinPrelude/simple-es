from abc import *

from torch import nn


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @abstractmethod
    def zero_init(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_param_list(self):
        pass

    @abstractmethod
    def apply_param(self, param_lst: list):
        pass
