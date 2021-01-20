import random
from copy import deepcopy

import cv2
import gym
import numpy as np
from gym import spaces
from PIL import Image


class EatApple:
    def __init__(
        self, world_size=20, reward_num=10, view_size=5, agent_num=2, random_goal=True
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
        self.agent_pos_list = []

        self.apple_pos = [
            (17, 12),
            (2, 10),
            (2, 10),
            (14, 3),
            (5, 7),
            (11, 5),
            (6, 12),
            (5, 15),
            (7, 3),
            (3, 10),
        ]
        self.init_agent_pos_list = [np.array([3, 3]), np.array([3, 4])]
        self.current_reward_num = self.total_reward_num
        self.max_step = 500
        self.current_step = 0

        self.floor_color = 1
        self.agent_color = 126
        self.apple_color = 250

    def generate_apple(self):
        reward_gen_size = self.world_size - (self.view_size // 2) * 2
        rand_pos_1d = [
            random.randint(0, pow(reward_gen_size, 2) - 1)
            for _ in range(self.total_reward_num)
        ]
        pos_2d = []
        pad = self.view_size // 2
        for pos in rand_pos_1d:
            pos_2d.append(
                ((pos // reward_gen_size) + pad, (pos % reward_gen_size) + pad)
            )

        return pos_2d

    def get_agent_views(self):
        agent_view_list = []
        for i in range(self.agent_num):
            p1 = self.agent_pos_list[i] - np.array(
                [self.view_size // 2, self.view_size // 2]
            )
            p2 = self.agent_pos_list[i] + np.array(
                [(self.view_size // 2) + 1, (self.view_size // 2) + 1]
            )
            agent_view_list.append(
                self.world[p1[0] : p2[0], p1[1] : p2[1]][np.newaxis, ...] / 255
            )
        return agent_view_list

    def reset(self):
        self.world = np.ones([self.world_size, self.world_size]) * self.floor_color
        assert (
            self.agent_num == 2 and not self.random_goal
        ), "fixed goal only support 2 agents."
        if self.random_goal:
            # reset agent positions.
            # TODO: Handle agent position overlap.
            for i in range(self.agent_num):
                tmp_pos = 0
                pos_min = self.view_size // 2
                pos_max = self.world_size - (self.view_size // 2)
                tmp_pos = np.random.randint(pos_min, pos_max, (2))  # 2 for x, y
                assert isinstance(
                    tmp_pos, np.ndarray
                ), "agent can't get place. try widen the map."
                self.agent_pos_list.append(tmp_pos)
        else:
            self.agent_pos_list = deepcopy(self.init_agent_pos_list)
        for agent_pos in self.agent_pos_list:
            self.world[agent_pos[0], agent_pos[1]] = self.agent_color

        self.current_reward_num = 10
        self.current_step = 0

        # generate apple
        if self.random_goal:
            self.apple_pos = self.generate_apple()
        for pos in self.apple_pos:
            self.world[pos] = self.apple_color
        return self.get_agent_views()

    def step(self, actions):
        done = False
        reward = 0
        action_dict = {
            0: np.array([-1, 0]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([0, 1]),
        }
        for i, act in enumerate(actions):
            act = int(act)
            curr_pos = self.agent_pos_list[i]
            tmp_moved_pos = curr_pos + action_dict[act]
            if self.world[tmp_moved_pos[0], tmp_moved_pos[1]] == self.agent_color:
                continue
            half_view = self.view_size // 2

            tmp_moved_pos[0] = max(tmp_moved_pos[0], half_view)
            tmp_moved_pos[0] = min(tmp_moved_pos[0], self.world_size - 1 - half_view)

            tmp_moved_pos[1] = max(tmp_moved_pos[1], half_view)
            tmp_moved_pos[1] = min(tmp_moved_pos[1], self.world_size - 1 - half_view)
            self.world[curr_pos[0], curr_pos[1]] = self.floor_color
            self.world[tmp_moved_pos[0], tmp_moved_pos[1]] = self.agent_color
            self.agent_pos_list[i] = tmp_moved_pos

        cur_r_num = np.where(self.world.flatten() == self.apple_color)[0]
        reward = self.current_reward_num - cur_r_num.size
        self.current_reward_num = cur_r_num.size
        self.current_step += 1
        if self.current_reward_num == 0 or self.current_step >= self.max_step:
            done = True
            self.reset()

        return (
            self.get_agent_views(),
            reward,
            done,
        )

    def render(self, mode="human"):
        render = self.world.copy()
        render = render.astype(np.uint8)
        render_img = Image.fromarray(render)
        render_img = render_img.resize((200, 200))
        render_img = np.asarray(render_img)
        cv2.imshow("image", render_img)
        views = self.get_agent_views()
        for i in range(self.agent_num):
            view = np.squeeze((views[i] * 255).astype(np.uint8))
            view = Image.fromarray(view)
            view = view.resize((100, 100))
            view = np.asarray(view)
            cv2.imshow(f"agent{i}", view)
        cv2.waitKey(30)


if __name__ == "__main__":
    for _ in range(100):
        agent_num = 2
        env = EatApple(random_goal=False, agent_num=agent_num)
        s = env.reset()
        d = False
        ep_r = 0
        t = 0
        while not d:
            t += 1
            env.render()
            actions = [random.randint(0, 3) for _ in range(agent_num)]
            # actions[0] = int(input())
            s, r, d = env.step(actions)
            ep_r += r
        print("reward: ", ep_r)
