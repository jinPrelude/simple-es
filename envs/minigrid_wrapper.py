from envs.atari_wrapper import ImageToPyTorch
import numpy as np
import gym
import pybullet_envs
from gym_minigrid.wrappers import ImgObsWrapper, FullyObsWrapper

gym.logger.set_level(40)


class MinigridWrapper:
    def __init__(self, name, max_step=None, pomdp=False):
        self.env = gym.make(name)
        if not pomdp:
            self.env = FullyObsWrapper(self.env)
        self.env = ImgObsWrapper(self.env)
        self.max_step = max_step
        self.curr_step = 0
        self.name = name

    def reset(self):
        self.curr_step = 0
        return_list = {}
        transition = {}
        s = self.env.reset()
        s = np.transpose(s, (-1, 0, 1))
        transition["state"] = s
        return_list["0"] = transition
        return return_list

    def step(self, action):
        self.curr_step += 1
        return_list = {}
        transition = {}
        s, r, d, info = self.env.step(action["0"])
        s = np.transpose(s, (-1, 0, 1))
        if self.max_step != "None":
            if self.curr_step >= self.max_step or d:
                d = True
        transition["state"] = s
        transition["reward"] = r
        transition["done"] = d
        transition["info"] = info
        return_list["0"] = transition
        return return_list, r, d, info

    def get_agent_ids(self):
        return ["0"]

    def render(self, mode):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()
