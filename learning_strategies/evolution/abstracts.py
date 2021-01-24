from abc import *


class BaseESLoop(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, env, network, cpu_num):
        self.env = env
        self.network = network
        self.cpu_num = cpu_num

    @abstractmethod
    def run(self):
        pass
