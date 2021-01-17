import os
import time
from copy import deepcopy

import numpy as np
import ray
import torch

from envs import EatApple
from networks import EatAppleModel
from utils import slice_list

from .base import BaseLS


@ray.remote
class RolloutWorker:
    def __init__(self, env_name, offspring_id, worker_id):
        os.environ["MKL_NUM_THREADS"] = "1"
        if env_name == "EatApple":
            self.env = EatApple(random_goal=False)
        self.groups = offspring_id[worker_id]
        self.worker_id = worker_id

    def rollout(self):
        rewards = []
        for i, (model1, model2) in enumerate(self.groups):
            (n1, n2) = self.env.reset()
            n1 = torch.from_numpy(n1[np.newaxis, ...]).float()
            n2 = torch.from_numpy(n2[np.newaxis, ...]).float()
            hidden1 = model1.init_hidden()
            hidden2 = model2.init_hidden()
            done = False
            episode_reward = 0
            while not done:
                with torch.no_grad():
                    # ray.util.pdb.set_trace()
                    action1, hidden1 = model1(n1, hidden1)
                    action2, hidden2 = model2(n2, hidden2)
                action1 = torch.argmax(action1).detach().numpy()
                action2 = torch.argmax(action2).detach().numpy()
                (n1, n2), r, done = self.env.step([action1, action2])
                n1 = torch.from_numpy(n1[np.newaxis, ...]).float()
                n2 = torch.from_numpy(n2[np.newaxis, ...]).float()
                # self.env.render()
                episode_reward += r
            rewards.append([(self.worker_id, i), episode_reward])
        return rewards


def gen_offspring_group(
    group: list, sigma: object, group_num: int, agents_num_per_group: int
) -> list:
    groups = []
    for _ in range(group_num):
        child_group = []
        for agent in group:
            tmp_agent = deepcopy(agent)
            for param in tmp_agent.parameters():
                with torch.no_grad():
                    noise = torch.normal(0, sigma, size=param.size())
                    param.add_(noise)

            child_group.append(tmp_agent)
        groups.append(child_group)
    return groups


class Gaussian(BaseLS):
    def __init__(self, env_name, network, cpu_num, group_num=363):
        super().__init__(env_name, cpu_num)
        if network == "EatAppleModel":
            self.network = EatAppleModel()
            self.network.init_weights(0, 1e-7)
        self.group_num = group_num
        self.elite_num = group_num // 10
        self.elite_models = [
            [self.network for _ in range(2)] for _ in range(self.elite_num)
        ]

        self.init_sigma = 1
        self.sigma_decay = 0.995

        ray.init()

    def run(self):

        curr_sigma = self.init_sigma
        start_time = time.time()
        ep_num = 0
        while True:
            ep_num += 1
            if self.cpu_num == 1:
                offspring_array = []
                for p in self.elite_models:
                    offspring_array[0:0] = gen_offspring_group(
                        p,
                        sigma=curr_sigma,
                        group_num=self.group_num // self.elite_num,
                        agents_num_per_group=2,
                    )
                offspring_array = slice_list(offspring_array, self.cpu_num)
                rollout_worker = RolloutWorker(self.env_name, offspring_array, 0)
                rewards = rollout_worker.rollout()
                rewards = sorted(rewards, key=lambda l: l[1], reverse=True)
                elite_ids = rewards[: self.elite_num]
                self.elite_models = []
                for id in elite_ids:
                    self.elite_models.append(offspring_array[id[0][0]][id[0][1]])
                print("Best reward :", rewards[0][1])
            else:
                offspring_array = []
                for p in self.elite_models:
                    offspring_array.append(p)
                    offspring_array[0:0] = gen_offspring_group(
                        p,
                        sigma=curr_sigma,
                        group_num=(self.group_num // self.elite_num) - 1,
                        agents_num_per_group=2,
                    )
                offspring_array = slice_list(offspring_array, self.cpu_num)
                offspring_id = ray.put(offspring_array)
                actors = [
                    RolloutWorker.remote(self.env_name, offspring_id, worker_id)
                    for worker_id in range(self.cpu_num)
                ]
                rollout_ids = [actor.rollout.remote() for actor in actors]
                rewards = []
                while len(rollout_ids):
                    done_id, rollout_ids = ray.wait(rollout_ids)
                    output = ray.get(done_id)
                    for li in output:
                        rewards[0:0] = li  # fast way to concatenate lists
                rewards = sorted(rewards, key=lambda l: l[1], reverse=True)
                elite_ids = rewards[: self.elite_num]
                self.elite_models = []
                for id in elite_ids:
                    self.elite_models.append(offspring_array[id[0][0]][id[0][1]])
                del offspring_id
            consumed_time = time.time() - start_time
            print(
                f"episode: {ep_num}, Best reward  {rewards[0][1]}, sigma: {curr_sigma:.3f}, time: {int(consumed_time)}"
            )
            save_dir = "saved_models/" + f"ep_{ep_num}/"
            os.makedirs(save_dir)
            torch.save(self.elite_models[0][0].state_dict(), save_dir + "agent1")
            torch.save(self.elite_models[0][1].state_dict(), save_dir + "agent2")
            rewards = []
            curr_sigma *= self.sigma_decay
