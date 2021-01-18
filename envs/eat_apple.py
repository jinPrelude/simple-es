import random

import cv2
import gym
import numpy as np
from gym import spaces


class EatApple(gym.Env):
    def __init__(self, random_goal=True):
        self.random_goal = random_goal
        self.world_size = 20
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

    def reset(self):
        self.world = np.ones([self.world_size, self.world_size]) * self.floor_color
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
        self.world[self.agent1_pos[0], self.agent1_pos[1]] = self.agent_color
        self.world[self.agent2_pos[0], self.agent2_pos[1]] = self.agent_color

        # generate apple
        if self.random_goal:
            self.apple_pos = self.generate_apple()
        for pos in self.apple_pos:
            self.world[pos] = self.apple_color

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
        return (agent1_view[np.newaxis, ...] / 255, agent2_view[np.newaxis, ...] / 255)

    def step(self, actions):
        done = False
        reward = 0
        if actions[0] == 0:  # up
            if not (self.agent1_pos - np.array([1, 0]) == self.agent2_pos).all():
                self.world[self.agent1_pos[0], self.agent1_pos[1]] = self.floor_color
                self.agent1_pos[0] = max(self.agent1_pos[0] - 1, self.view_size // 2)
                self.world[self.agent1_pos[0], self.agent1_pos[1]] = self.agent_color
        elif actions[0] == 1:  # down
            if not (self.agent1_pos + np.array([1, 0]) == self.agent2_pos).all():
                self.world[self.agent1_pos[0], self.agent1_pos[1]] = self.floor_color
                self.agent1_pos[0] = min(
                    self.agent1_pos[0] + 1, self.world_size - 1 - (self.view_size // 2)
                )
                self.world[self.agent1_pos[0], self.agent1_pos[1]] = self.agent_color
        elif actions[0] == 2:  # left
            if not (self.agent1_pos - np.array([0, 1]) == self.agent2_pos).all():
                self.world[self.agent1_pos[0], self.agent1_pos[1]] = self.floor_color
                self.agent1_pos[1] = max(self.agent1_pos[1] - 1, self.view_size // 2)
                self.world[self.agent1_pos[0], self.agent1_pos[1]] = self.agent_color
        elif actions[0] == 3:  # right
            if not (self.agent1_pos + np.array([0, 1]) == self.agent2_pos).all():
                self.world[self.agent1_pos[0], self.agent1_pos[1]] = self.floor_color
                self.agent1_pos[1] = min(
                    self.agent1_pos[1] + 1, self.world_size - 1 - (self.view_size // 2)
                )
                self.world[self.agent1_pos[0], self.agent1_pos[1]] = self.agent_color

        if actions[1] == 0:  # up
            if not (self.agent2_pos - np.array([1, 0]) == self.agent1_pos).all():
                self.world[self.agent2_pos[0], self.agent2_pos[1]] = self.floor_color
                self.agent2_pos[0] = max(self.agent2_pos[0] - 1, self.view_size // 2)
                self.world[self.agent2_pos[0], self.agent2_pos[1]] = self.agent_color
        elif actions[1] == 1:  # down
            if not (self.agent2_pos + np.array([1, 0]) == self.agent1_pos).all():
                self.world[self.agent2_pos[0], self.agent2_pos[1]] = self.floor_color
                self.agent2_pos[0] = min(
                    self.agent2_pos[0] + 1, self.world_size - 1 - (self.view_size // 2)
                )
                self.world[self.agent2_pos[0], self.agent2_pos[1]] = self.agent_color
        elif actions[1] == 2:  # left
            if not (self.agent2_pos - np.array([0, 1]) == self.agent1_pos).all():
                self.world[self.agent2_pos[0], self.agent2_pos[1]] = self.floor_color
                self.agent2_pos[1] = max(self.agent2_pos[1] - 1, self.view_size // 2)
                self.world[self.agent2_pos[0], self.agent2_pos[1]] = self.agent_color
        elif actions[1] == 3:  # right
            if not (self.agent2_pos + np.array([0, 1]) == self.agent1_pos).all():
                self.world[self.agent2_pos[0], self.agent2_pos[1]] = self.floor_color
                self.agent2_pos[1] = min(
                    self.agent2_pos[1] + 1, self.world_size - 1 - (self.view_size // 2)
                )
                self.world[self.agent2_pos[0], self.agent2_pos[1]] = self.agent_color

        cur_r_num = np.where(self.world.flatten() == self.apple_color)[0]
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
            (agent1_view[np.newaxis, ...] / 255, agent2_view[np.newaxis, ...] / 255),
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
        cv2.waitKey(30)


if __name__ == "__main__":
    for _ in range(100):
        env = EatApple(random_goal=False)
        s = env.reset()
        d = False
        ep_r = 0
        while not d:
            env.render()
            # action1 = int(input())
            action1 = random.randint(0, 3)
            s, r, d = env.step([random.randint(0, 3), action1])
            ep_r += r
        print("reward: ", ep_r)
