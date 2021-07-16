import gym
import pybullet_envs
from .atari_wrapper import atari_env_generator

gym.logger.set_level(40)


class GymWrapper:
    def __init__(self, name, max_step=None, pomdp=False):
        self.env = gym.make(name)
        if pomdp:
            if "LunarLander" in name:
                print("POMDP LunarLander")
                self.env = LunarLanderPOMDP(self.env)
            elif "CartPole" in name:
                print("POMDP CartPole")
                self.env = CartPolePOMDP(self.env)
            else:
                raise AssertionError(f"{name} doesn't support POMDP.")
        if "Pong" in name:
            self.env = atari_env_generator(name)
        self.max_step = max_step
        self.curr_step = 0
        self.name = name

    def reset(self):
        self.curr_step = 0
        return_list = {}
        transition = {}
        s = self.env.reset()
        transition["state"] = s
        return_list["0"] = transition
        return return_list

    def step(self, action):
        self.curr_step += 1
        return_list = {}
        transition = {}
        s, r, d, info = self.env.step(action["0"])
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

    def render(self):
        return self.env.render(mode="rgb_array")

    def close(self):
        self.env.close()


class LunarLanderPOMDP(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        # modify obs
        obs[2] = 0
        obs[3] = 0
        obs[5] = 0
        return obs


class CartPolePOMDP(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        # modify obs
        obs[1] = 0
        obs[3] = 0
        return obs
