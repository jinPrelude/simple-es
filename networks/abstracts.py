from abc import *

from torch import nn


class BaseNetwork(nn.Module, object):
    @abstractmethod
    def __init__(self, rollout_worker):
        super(BaseNetwork, self).__init__()
        self.rollout_worker = rollout_worker

    @abstractmethod
    def init_weights(self):
        pass


class BaseRNNNetwork(BaseNetwork):
    @abstractmethod
    def init_hidden(self):
        pass
