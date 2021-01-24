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
    def __init__(self, env, offspring_id, worker_id, eval_ep_num=10):
        os.environ["MKL_NUM_THREADS"] = "1"
        self.env = env
        self.groups = offspring_id[worker_id]
        self.worker_id = worker_id
        self.eval_ep_num = eval_ep_num

    def rollout(self):
        rewards = []
        for i, models in enumerate(self.groups):
            total_reward = 0
            for _ in range(self.eval_ep_num):
                states = self.env.reset()
                hidden_states = {}
                done = False
                for k, model in models.items():
                    hidden_states[k] = model.init_hidden()
                while not done:
                    actions = {}
                    with torch.no_grad():
                        # ray.util.pdb.set_trace()
                        for k, model in models.items():
                            s = torch.from_numpy(
                                states[k]["state"][np.newaxis, ...]
                            ).float()
                            a, hidden_states[k] = model(s, hidden_states[k])
                            actions[k] = torch.argmax(a).detach().numpy()
                    states, r, done, info = self.env.step(actions)
                    # self.env.render()
                    total_reward += r
            rewards.append([(self.worker_id, i), total_reward / self.eval_ep_num])
        return rewards


class simple_gaussian_offspring:
    def __init__(self, init_sigma, sigma_decay, elite_ratio, group_num):
        self.init_sigma = init_sigma
        self.sigma_decay = sigma_decay
        self.elite_ratio = elite_ratio
        self.group_num = group_num
        self.elite_models = []

        self.curr_sigma = self.init_sigma
        self.elite_num = max(1, int(group_num * elite_ratio))

    @staticmethod
    def _gen_mutation(group: dict, sigma: object, group_num: int):
        offsprings_group = []
        for _ in range(group_num):
            agent_group = {}
            for agent_id, agent in group.items():
                tmp_agent = deepcopy(agent)
                for param in tmp_agent.parameters():
                    with torch.no_grad():
                        noise = torch.normal(0, sigma, size=param.size())
                        param.add_(noise)

                agent_group[agent_id] = tmp_agent
            offsprings_group.append(agent_group)
        return offsprings_group

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

    def init_offspring(self, network, agent_ids):
        network.init_weights(0, 1e-7)
        for _ in range(self.elite_num):
            group = {}
            for agent_id in agent_ids:
                group[agent_id] = network
            self.elite_models.append(group)
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
        ray.init()

    def run(self):
        if self.cpu_num <= 1:
            self.debug_mode()

        # init offsprings
        offsprings = self.offspring_strategy.init_offspring(
            self.network, self.env.get_agent_ids()
        )
        offsprings = slice_list(offsprings, self.cpu_num)

        prev_reward = float("-inf")
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
                f"episode: {ep_num}, Best reward: {best_reward:.2f}, sigma: {curr_sigma:.3f}, time: {consumed_time:.2f}, rollout_t: {rollout_consumed_time:.2f}, eval_t: {eval_consumed_time:.2f}"
            )

            prev_reward = best_reward
            save_dir = "saved_models/" + f"ep_{ep_num}/"
            os.makedirs(save_dir)
            elite_group = self.offspring_strategy.get_elite_models()[0]
            for k, model in elite_group.items():
                torch.save(model.state_dict(), save_dir + f"{k}")

    def debug_mode(self):
        print(
            "You have entered debug mode. Don't forget to detatch ray.remote() of the rollout worker."
        )
        # init offsprings
        offsprings = self.offspring_strategy.init_offspring(
            self.network, self.env.get_agent_ids()
        )
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
                f"episode: {ep_num}, Best reward: {best_reward:.2f}, sigma: {curr_sigma:.3f}, time: {consumed_time:.2f}, rollout_t: {rollout_consumed_time:.2f}, eval_t: {eval_consumed_time:.2f}"
            )
