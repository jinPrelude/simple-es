import os
from abc import *


class BaseESLoop(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass


class BaseOffspringStrategy(metaclass=ABCMeta):
    @abstractmethod
    def __init__(selfm):
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
