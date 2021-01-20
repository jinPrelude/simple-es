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
        self, init_sigma, sigma_decay, elite_ratio, group_num, agent_per_group=None
    ):
        self.init_sigma = init_sigma
        self.sigma_decay = sigma_decay
        self.elite_ratio = elite_ratio
        self.group_num = group_num
        self.agent_per_group = agent_per_group
        self.elite_models = []

        self.curr_sigma = self.init_sigma
        self.elite_num = max(1, int(group_num * elite_ratio))

    @staticmethod
    def _gen_mutation(group: list, sigma: object, group_num: int):
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

    def _gen_offsprings(self):
        offspring_array = []
        for p in self.elite_models:
            offspring_array.append(p)
            offspring_array[0:0] = self._gen_mutation(
                p,
                sigma=self.curr_sigma,
                group_num=(self.group_num // self.elite_num) - 1,
            )
        return offspring_array

    def get_elite_models(self):
        return self.elite_models

    def set_agent_per_group(self, agent_per_group):
        self.agent_per_group = agent_per_group

    def init_offspring(self, network):
        assert self.agent_per_group, "Call agent_per_group() before initialize."
        network.init_weights(0, 1e-7)

        self.elite_models = [
            [network for _ in range(self.agent_per_group)]
            for _ in range(self.elite_num)
        ]
        return self._gen_offsprings()

    def evaluate(self, result, offsprings):
        results = sorted(result, key=lambda l: l[1], reverse=True)
        best_reward = results[0][1]
        elite_ids = results[: self.elite_num]
        self.elite_models = []
        for id in elite_ids:
            self.elite_models.append(offsprings[id[0][0]][id[0][1]])
        offsprings = self._gen_offsprings()
        self.curr_sigma *= self.sigma_decay
        return offsprings, best_reward, self.curr_sigma


class Gaussian(BaseLS):
    def __init__(self, offspring_strategy, env, network, cpu_num):
        super().__init__(env, network, cpu_num)
        self.network.init_weights(0, 1e-7)
        self.env = env
        self.offspring_strategy = offspring_strategy
        self.offspring_strategy.set_agent_per_group(self.env.agent_num)
        ray.init()

    def run(self):
        if self.cpu_num <= 1:
            self.debug_mode()

        # init offsprings
        offsprings = self.offspring_strategy.init_offspring(self.network)
        offsprings = slice_list(offsprings, self.cpu_num)

        ep_num = 0
        while True:
            start_time = time.time()
            ep_num += 1

            # ray.put() offsprings & env
            offspring_id = ray.put(offsprings)
            env_id = ray.put(self.env)

            # create an actor by the number of cores
            actors = [
                RnnRolloutWorker.remote(env_id, offspring_id, worker_id)
                for worker_id in range(self.cpu_num)
            ]

            # start ollout actors
            rollout_start_time = time.time()
            rollout_ids = [actor.rollout.remote() for actor in actors]

            # wait until all the actors are finished
            results = []
            while len(rollout_ids):
                done_id, rollout_ids = ray.wait(rollout_ids)
                output = ray.get(done_id)
                for li in output:
                    results[0:0] = li  # fast way to concatenate lists
            rollout_consumed_time = time.time() - rollout_start_time

            # Offspring evaluation
            del offspring_id
            eval_start_time = time.time()
            offsprings, best_reward, curr_sigma = self.offspring_strategy.evaluate(
                results, offsprings
            )
            offsprings = slice_list(offsprings, self.cpu_num)
            eval_consumed_time = time.time() - eval_start_time

            # print log
            consumed_time = time.time() - start_time
            print(
                f"episode: {ep_num}, Best reward: {best_reward}, sigma: {curr_sigma:.3f}, time: {consumed_time:.2f}, rollout_t: {rollout_consumed_time:.2f}, eval_t: {eval_consumed_time:.2f}"
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
        # init offsprings
        offsprings = self.offspring_strategy.init_offspring(self.network)
        offsprings = slice_list(offsprings, self.cpu_num)

        ep_num = 0
        # start rollout
        while True:
            start_time = time.time()
            ep_num += 1

            rollout_start_time = time.time()
            rollout_worker = RnnRolloutWorker(self.env, offsprings, 0)
            results = rollout_worker.rollout()
            rollout_consumed_time = time.time() - rollout_start_time

            # Offspring evaluation
            eval_start_time = time.time()
            offsprings, best_reward, curr_sigma = self.offspring_strategy.evaluate(
                results, offsprings
            )
            offsprings = slice_list(offsprings, self.cpu_num)
            eval_consumed_time = time.time() - eval_start_time

            # print log
            consumed_time = time.time() - start_time
            print(
                f"episode: {ep_num}, Best reward: {best_reward}, sigma: {curr_sigma:.3f}, time: {consumed_time:.2f}, rollout_t: {rollout_consumed_time:.2f}, eval_t: {eval_consumed_time:.2f}"
            )
