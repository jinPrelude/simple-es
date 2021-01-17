import random

import cv2
import gym
import numpy as np
from gym import spaces
from PIL import Image


class EatApple(gym.Env):
    def __init__(self):
        self.world_size = 30
        self.total_reward_num = 10
        self.view_size = 5
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 3]), dtype=int
        )
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(2, self.world_size, self.world_size), dtype=np.int8
        )
        self.world = np.zeros([self.world_size, self.world_size])
        self.agent1_pos = np.array([self.view_size // 2, self.view_size // 2])
        self.agent2_pos = np.array([self.world_size - 2, self.world_size - 2])
        self.reward_pos = []
        self.current_reward_num = self.total_reward_num
        self.max_step = 500
        self.current_step = 0

    def generate_apple(self):
        reward_gen_size = self.world_size - (self.view_size // 2) * 2
        rand_pos_1d = [
            random.randint(0, pow(reward_gen_size, 2) - 1)
            for _ in range(self.total_reward_num)
        ]

        pad = self.view_size // 2
        for pos in rand_pos_1d:
            pos_2d = [(pos // reward_gen_size) + pad, (pos % reward_gen_size) + pad]

            self.world[pos_2d[0], pos_2d[1]] = 2

    def reset(self):
        self.world = np.zeros([self.world_size, self.world_size])
        self.agent1_pos = np.array([self.view_size // 2, self.view_size // 2])
        self.agent2_pos = np.array(
            [
                self.world_size - 1 - self.view_size // 2,
                self.world_size - 1 - self.view_size // 2,
            ]
        )
        self.reward_pos = []
        self.current_reward_num = 10
        self.current_step = 0
        self.world[self.agent1_pos[0], self.agent1_pos[1]] = 1
        self.world[self.agent2_pos[0], self.agent2_pos[1]] = 1
        self.generate_apple()
        agent1_view = self.world[
            self.agent1_pos[0]
            - (self.view_size // 2) : self.agent1_pos[0]
            + 1
            + (self.view_size // 2),
            self.agent1_pos[0]
            - (self.view_size // 2) : self.agent1_pos[0]
            + 1
            + (self.view_size // 2),
        ]
        agent2_view = self.world[
            self.agent2_pos[0]
            - (self.view_size // 2) : self.agent2_pos[0]
            + 1
            + (self.view_size // 2),
            self.agent2_pos[0]
            - (self.view_size // 2) : self.agent2_pos[0]
            + 1
            + (self.view_size // 2),
        ]
        return (agent1_view[np.newaxis, ...], agent2_view[np.newaxis, ...])

    def step(self, actions):
        done = False
        reward = 0
        if actions[0] == 0:  # up
            if not (self.agent1_pos - np.array([1, 0]) == self.agent2_pos).all():
                self.world[self.agent1_pos[0], self.agent1_pos[1]] = 0
                self.agent1_pos[0] = max(self.agent1_pos[0] - 1, self.view_size // 2)
                self.world[self.agent1_pos[0], self.agent1_pos[1]] = 1
        elif actions[0] == 1:  # down
            if not (self.agent1_pos + np.array([1, 0]) == self.agent2_pos).all():
                self.world[self.agent1_pos[0], self.agent1_pos[1]] = 0
                self.agent1_pos[0] = min(
                    self.agent1_pos[0] + 1, self.world_size - 1 - (self.view_size // 2)
                )
                self.world[self.agent1_pos[0], self.agent1_pos[1]] = 1
        elif actions[0] == 2:  # left
            if not (self.agent1_pos - np.array([0, 1]) == self.agent2_pos).all():
                self.world[self.agent1_pos[0], self.agent1_pos[1]] = 0
                self.agent1_pos[1] = max(self.agent1_pos[1] - 1, self.view_size // 2)
                self.world[self.agent1_pos[0], self.agent1_pos[1]] = 1
        elif actions[0] == 3:  # right
            if not (self.agent1_pos + np.array([0, 1]) == self.agent2_pos).all():
                self.world[self.agent1_pos[0], self.agent1_pos[1]] = 0
                self.agent1_pos[1] = min(
                    self.agent1_pos[1] + 1, self.world_size - 1 - (self.view_size // 2)
                )
                self.world[self.agent1_pos[0], self.agent1_pos[1]] = 1

        if actions[1] == 0:  # up
            if not (self.agent2_pos - np.array([1, 0]) == self.agent1_pos).all():
                self.world[self.agent2_pos[0], self.agent2_pos[1]] = 0
                self.agent2_pos[0] = max(self.agent2_pos[0] - 1, self.view_size // 2)
                self.world[self.agent2_pos[0], self.agent2_pos[1]] = 1
        elif actions[1] == 1:  # down
            if not (self.agent2_pos + np.array([1, 0]) == self.agent1_pos).all():
                self.world[self.agent2_pos[0], self.agent2_pos[1]] = 0
                self.agent2_pos[0] = min(
                    self.agent2_pos[0] + 1, self.world_size - 1 - (self.view_size // 2)
                )
                self.world[self.agent2_pos[0], self.agent2_pos[1]] = 1
        elif actions[1] == 2:  # left
            if not (self.agent2_pos - np.array([0, 1]) == self.agent1_pos).all():
                self.world[self.agent2_pos[0], self.agent2_pos[1]] = 0
                self.agent2_pos[1] = max(self.agent2_pos[1] - 1, self.view_size // 2)
                self.world[self.agent2_pos[0], self.agent2_pos[1]] = 1
        elif actions[1] == 3:  # right
            if not (self.agent2_pos + np.array([0, 1]) == self.agent1_pos).all():
                self.world[self.agent2_pos[0], self.agent2_pos[1]] = 0
                self.agent2_pos[1] = min(
                    self.agent2_pos[1] + 1, self.world_size - 1 - (self.view_size // 2)
                )
                self.world[self.agent2_pos[0], self.agent2_pos[1]] = 1

        cur_r_num = np.where(self.world.flatten() == 2)[0]
        reward = self.current_reward_num - cur_r_num.size
        self.current_reward_num = cur_r_num.size
        self.current_step += 1
        if self.current_reward_num == 0 or self.current_step >= self.max_step:
            done = True
            self.reset()
        agent1_view = self.world[
            self.agent1_pos[0]
            - (self.view_size // 2) : self.agent1_pos[0]
            + 1
            + (self.view_size // 2),
            self.agent1_pos[0]
            - (self.view_size // 2) : self.agent1_pos[0]
            + 1
            + (self.view_size // 2),
        ]
        agent2_view = self.world[
            self.agent2_pos[0]
            - (self.view_size // 2) : self.agent2_pos[0]
            + 1
            + (self.view_size // 2),
            self.agent2_pos[0]
            - (self.view_size // 2) : self.agent2_pos[0]
            + 1
            + (self.view_size // 2),
        ]
        return (
            (agent1_view[np.newaxis, ...], agent2_view[np.newaxis, ...]),
            reward,
            done,
        )

    def render(self, mode="human"):
        render = self.world.copy() * 120
        render = render.astype(np.uint8)
        render_img = Image.fromarray(render)
        render_img = render_img.resize((200, 200))
        render_img = np.asarray(render_img)
        cv2.imshow("image", render_img)
        cv2.waitKey(50)


if __name__ == "__main__":
    env = EatApple()
    for _ in range(100):
        s = env.reset()
        d = False
        ep_r = 0
        while not d:
            env.render()
            action1 = int(input())
            s, r, d = env.step([random.randint(0, 3), action1])
            ep_r += r
        print("reward: ", ep_r)
