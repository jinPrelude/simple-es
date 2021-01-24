import random
from copy import deepcopy

import cv2
import gym
import numpy as np
from gym import spaces
from PIL import Image


class EatApple:
    def __init__(
        self,
        world_size=20,
        reward_num=10,
        view_size=5,
        agent_num=2,
        random_goal=True,
        max_step=500,
    ):
        self.random_goal = random_goal
        self.world_size = world_size
        self.total_reward_num = reward_num
        self.view_size = view_size
        self.agent_num = agent_num
        self.action_space = spaces.Box(
            low=np.zeros(agent_num), high=np.ones(agent_num) * 3, dtype=int
        )
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(agent_num, self.world_size, self.world_size),
            dtype=np.float,
        )
        self.world = np.zeros([self.world_size, self.world_size])
        self.agent_pos_list = {}

        self.fixed_apple_pos = [130, 147, 178, 151, 28, 33, 116, 81, 160, 138]
        self.fixed_agent_pos = [112, 24]
        self.current_reward_num = self.total_reward_num
        self.max_step = max_step
        self.current_step = 0

        self.floor_color = 1
        self.agent_color = 126
        self.apple_color = 250

    def _init_world(self):
        pad = self.view_size // 2
        self.world = np.ones([self.world_size, self.world_size]) * self.floor_color
        if not self.random_goal and self.agent_num != 2:
            raise AssertionError("fixed goal only support 2 agents.")
        reward_gen_size = self.world_size - (self.view_size // 2) * 2
        pos_ids = [x for x in range(pow(reward_gen_size, 2) - 1)]
        pos_ids = random.sample(pos_ids, self.total_reward_num + self.agent_num)
        apple_pos_1d = pos_ids[: self.total_reward_num]
        agent_pos_1d = pos_ids[self.total_reward_num :]
        if not self.random_goal:
            apple_pos_1d = self.fixed_apple_pos
            agent_pos_1d = self.fixed_agent_pos
        for pos in apple_pos_1d:
            pos = ((pos // reward_gen_size) + pad, (pos % reward_gen_size) + pad)
            self.world[pos] = self.apple_color
        for i, pos in enumerate(agent_pos_1d):
            pos = ((pos // reward_gen_size) + pad, (pos % reward_gen_size) + pad)
            self.agent_pos_list[str(i)] = pos
            self.world[pos] = self.agent_color

    def get_agent_ids(self):
        return [str(x) for x in range(self.agent_num)]

    def get_agent_views(self, key):
        p1 = self.agent_pos_list[key] - np.array(
            [self.view_size // 2, self.view_size // 2]
        )
        p2 = self.agent_pos_list[key] + np.array(
            [(self.view_size // 2) + 1, (self.view_size // 2) + 1]
        )
        s = self.world[p1[0] : p2[0], p1[1] : p2[1]][np.newaxis, ...] / 255
        return s

    def reset(self):

        self._init_world()

        self.current_reward_num = 10
        self.current_step = 0

        return_list = {}
        for key, _ in self.agent_pos_list.items():
            return_list[key] = {"state": self.get_agent_views(key)}
        return return_list

    def step(self, actions):
        done = False
        reward = 0
        action_dict = {
            0: np.array([-1, 0]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([0, 1]),
        }
        return_dict = {}
        self.current_step += 1
        if self.current_reward_num == 0 or self.current_step >= self.max_step:
            done = True
        total_reward = 0
        for key, act in actions.items():
            transition = {}
            assert key in self.agent_pos_list.keys()
            act = int(act)
            curr_pos = self.agent_pos_list[key]
            tmp_moved_pos = curr_pos + action_dict[act]
            reward = 0
            if self.world[tmp_moved_pos[0], tmp_moved_pos[1]] != self.agent_color:
                half_view = self.view_size // 2

                tmp_moved_pos[0] = max(tmp_moved_pos[0], half_view)
                tmp_moved_pos[0] = min(
                    tmp_moved_pos[0], self.world_size - 1 - half_view
                )

                tmp_moved_pos[1] = max(tmp_moved_pos[1], half_view)
                tmp_moved_pos[1] = min(
                    tmp_moved_pos[1], self.world_size - 1 - half_view
                )
                self.world[curr_pos[0], curr_pos[1]] = self.floor_color
                self.world[tmp_moved_pos[0], tmp_moved_pos[1]] = self.agent_color
                self.agent_pos_list[key] = tmp_moved_pos
                cur_r_num = np.where(self.world.flatten() == self.apple_color)[0]
                reward = self.current_reward_num - cur_r_num.size
                total_reward += reward
                self.current_reward_num = cur_r_num.size
            transition["state"] = self.get_agent_views(key)
            transition["reward"] = reward
            transition["done"] = done
            transition["info"] = {}
            return_dict[key] = transition

        return return_dict, total_reward, done, {}

    def render(self, mode="human"):
        render = self.world.copy()
        render = render.astype(np.uint8)
        render_img = Image.fromarray(render)
        render_img = render_img.resize((200, 200))
        render_img = np.asarray(render_img)
        cv2.imshow("image", render_img)
        # views = self.get_agent_views()
        # for i in range(self.agent_num):
        #     view = np.squeeze((views[i] * 255).astype(np.uint8))
        #     view = Image.fromarray(view)
        #     view = view.resize((100, 100))
        #     view = np.asarray(view)
        #     cv2.imshow(f"agent{i}", view)
        cv2.waitKey(30)


if __name__ == "__main__":
    for _ in range(100):
        agent_num = 2
        env = EatApple(agent_num=agent_num, max_step=100)
        s = env.reset()
        agent_ids = s.keys()
        d = False
        ep_r = 0
        t = 0
        while not d:
            t += 1
            env.render()
            actions = {}
            for i in agent_ids:
                actions[i] = random.randint(0, 3)
            # actions[0] = int(input())
            obs, r, d, _ = env.step(actions)
            ep_r += r
        print("reward: ", ep_r)
