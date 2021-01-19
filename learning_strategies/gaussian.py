import os
import time
from copy import deepcopy

import numpy as np
import ray
import torch

from utils import slice_list

from .base import BaseLS


@ray.remote
class RnnRolloutWorker:
    def __init__(self, env, offspring_id, worker_id):
        os.environ["MKL_NUM_THREADS"] = "1"
        self.env = env
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


class simple_gaussian_offspring:
    def __init__(
        self, init_sigma, sigma_decay, elite_ratio, group_num, agent_per_group
    ):
        self.init_sigma = init_sigma
        self.sigma_decay = sigma_decay
        self.elite_ratio = elite_ratio
        self.group_num = group_num
        self.agent_per_group = agent_per_group
        self.elite_models = []

        self.curr_sigma = self.init_sigma
        self.elite_num = elite_num = int(group_num * elite_ratio)

    @staticmethod
    def gen_offspring_group(group: list, sigma: object, group_num: int):
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

    def get_elite_models(self):
        return self.elite_models

    def init_offspring(self, network):
        network.init_weights(0, 1e-7)

        self.elite_models = [
            [network for _ in range(self.agent_per_group)]
            for _ in range(self.elite_num)
        ]
        offspring_array = []
        for p in self.elite_models:
            offspring_array.append(p)
            offspring_array[0:0] = self.gen_offspring_group(
                p,
                sigma=self.curr_sigma,
                group_num=(self.group_num // self.elite_num) - 1,
            )
        return offspring_array

    def evaluate(self, result, offsprings):
        results = sorted(result, key=lambda l: l[1], reverse=True)
        best_reward = results[0][1]
        elite_ids = results[: self.elite_num]
        self.elite_models = []
        for id in elite_ids:
            self.elite_models.append(offsprings[id[0][0]][id[0][1]])
        offsprings = []
        for p in self.elite_models:
            offsprings.append(p)
            offsprings[0:0] = self.gen_offspring_group(
                p,
                sigma=self.curr_sigma,
                group_num=(self.group_num // self.elite_num) - 1,
            )
        self.curr_sigma *= self.sigma_decay
        return offsprings, best_reward, self.curr_sigma


class Gaussian(BaseLS):
    def __init__(
        self, env, network, cpu_num, group_num=480, elite_ratio=0.1, init_sigma=2
    ):
        super().__init__(env, network, cpu_num)
        self.network.init_weights(0, 1e-7)
        self.env = env
        self.agent_per_group = self.env.agent_per_group
        self.group_num = group_num
        self.offspring_strategy = simple_gaussian_offspring(
            init_sigma, 0.995, elite_ratio, group_num, self.agent_per_group
        )

        ray.init()

    def run(self):
        if self.cpu_num <= 1:
            self.debug_mode()
        start_time = time.time()
        offsprings = self.offspring_strategy.init_offspring(self.network)
        offsprings = slice_list(offsprings, self.cpu_num)

        ep_num = 0
        # start rollout
        while True:
            ep_num += 1

            offspring_id = ray.put(offsprings)
            # ray.put() env
            env_id = ray.put(self.env)

            # Create an actor by the number of cores
            actors = [
                RnnRolloutWorker.remote(env_id, offspring_id, worker_id)
                for worker_id in range(self.cpu_num)
            ]

            # Rollout actors
            rollout_ids = [actor.rollout.remote() for actor in actors]

            # Wait until all the actors are finished
            results = []
            while len(rollout_ids):
                done_id, rollout_ids = ray.wait(rollout_ids)
                output = ray.get(done_id)
                for li in output:
                    results[0:0] = li  # fast way to concatenate lists

            # Offspring evaluation
            del offspring_id
            offsprings, best_reward, curr_sigma = self.offspring_strategy.evaluate(
                results, offsprings
            )
            offsprings = slice_list(offsprings, self.cpu_num)

            # print log
            consumed_time = time.time() - start_time
            print(
                f"episode: {ep_num}, Best reward  {best_reward}, sigma: {curr_sigma:.3f}, time: {int(consumed_time)}"
            )

            # save elite model of the current episode.
            save_dir = "saved_models/" + f"ep_{ep_num}/"
            os.makedirs(save_dir)
            elite_group = self.offspring_strategy.get_elite_models()
            torch.save(elite_group[0][0].state_dict(), save_dir + "agent1")
            torch.save(elite_group[0][1].state_dict(), save_dir + "agent2")

    def debug_mode(self):
        print(
            "You have entered debug mode. Don't forget to detatch ray.remote() of the rollout worker."
        )
        curr_sigma = self.init_sigma
        start_time = time.time()
        ep_num = 0
        while True:
            offspring_array = []
            for p in self.elite_models:
                offspring_array[0:0] = gen_offspring_group(
                    p, sigma=curr_sigma, group_num=self.group_num // self.elite_num,
                )
            offspring_array = slice_list(offspring_array, self.cpu_num)
            rollout_worker = RnnRolloutWorker(self.env, offspring_array, 0)
            rewards = rollout_worker.rollout()
            rewards = sorted(rewards, key=lambda l: l[1], reverse=True)
            elite_ids = rewards[: self.elite_num]
            self.elite_models = []
            for id in elite_ids:
                self.elite_models.append(offspring_array[id[0][0]][id[0][1]])
            # print log
            consumed_time = time.time() - start_time
            print(
                f"episode: {ep_num}, Best reward  {rewards[0][1]}, sigma: {curr_sigma:.3f}, time: {int(consumed_time)}"
            )

            # save elite model of the current episode.
            save_dir = "saved_models/" + f"ep_{ep_num}/"
            os.makedirs(save_dir)
            torch.save(self.elite_models[0][0].state_dict(), save_dir + "agent1")
            torch.save(self.elite_models[0][1].state_dict(), save_dir + "agent2")

            # decay sigma
            curr_sigma *= self.sigma_decay
