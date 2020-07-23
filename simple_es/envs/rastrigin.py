import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import logger, spaces
from matplotlib import cm


class Rastrigin(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(Rastrigin, self).__init__()
        X = np.linspace(-5.12, 5.12, 100)
        Y = np.linspace(-5.12, 5.12, 100)
        self.X, self.Y = np.meshgrid(X, Y)

        self.action_space = spaces.Discrete(2)
        self.observation_space = None

    def _rastrigin(self, x, y):
        return (
            (x ** 2 - 10 * np.cos(2 * np.pi * x))
            + (y ** 2 - 10 * np.cos(2 * np.pi * y))
            + 20
        )

    def step(self, action):
        x = action[0]
        y = action[1]
        z = self._rastrigin(x, y)
        reward = -z
        return None, reward, True, None  # state, reward, done, info

    def reset(self):
        pass

    def render(self, mode="human"):
        pass

    def close(self):
        pass
