import os
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


class BaseOffspringStrategy(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, elite_num, offspring_num):
        self.elite_num = elite_num
        self.offspring_num = offspring_num
        self.elite_models = []

    @abstractmethod
    def _gen_mutation(self):
        pass

    @abstractmethod
    def _gen_offsprings(self):
        pass

    @abstractmethod
    def get_elite_model(self):
        pass

    @abstractmethod
    def init_offspring(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

