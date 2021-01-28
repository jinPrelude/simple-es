import os
import time
from copy import deepcopy

import numpy as np
import ray
import torch
from hydra.utils import instantiate

from utils import slice_list

from .abstracts import BaseESLoop
from .rollout_workers import RNNRolloutWorker


class ESLoop(BaseESLoop):
    def __init__(self, logger, offspring_strategy, env, network, cpu_num):
        super(ESLoop, self).__init__(env, network, cpu_num)
        self.network.init_weights(0, 1e-7)
        self.logger = logger
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
            rolloutworker = self.network.rollout_worker
            actors = [
                rolloutworker.remote(env_id, offspring_id, worker_id)
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
            self.logger.info(
                f"episode: {ep_num}, Best reward: {best_reward:.2f}, sigma: {curr_sigma:.3f}, time: {consumed_time:.2f}, rollout_t: {rollout_consumed_time:.2f}, eval_t: {eval_consumed_time:.2f}"
            )

            prev_reward = best_reward
            save_dir = "saved_models/" + f"ep_{ep_num}/"
            os.makedirs(save_dir)
            elite_group = self.offspring_strategy.get_elite_model()
            for k, model in elite_group.items():
                torch.save(model.state_dict(), save_dir + f"{k}")

    def debug_mode(self):
        self.logger.info(
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
            rollout_worker = self.network.rollout_worker
            rollout_worker = rollout_worker(self.env, offsprings, 0)
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
            self.logger.info(
                f"episode: {ep_num}, Best reward: {best_reward:.2f}, sigma: {curr_sigma:.3f}, time: {consumed_time:.2f}, rollout_t: {rollout_consumed_time:.2f}, eval_t: {eval_consumed_time:.2f}"
            )
