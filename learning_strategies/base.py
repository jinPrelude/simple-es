from abc import *


class BaseLS(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, env_name, cpu_num):
        self.env_name = env_name
        self.cpu_num = cpu_num

    @abstractmethod
    def run(self):
        pass
