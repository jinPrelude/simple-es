import os
import time
from datetime import datetime
from collections import deque
from copy import deepcopy

import numpy as np
import multiprocessing as mp
import torch

import wandb
from utils import slice_list

from .abstracts import BaseESLoop


class ESLoop(BaseESLoop):
    def __init__(
        self,
        offspring_strategy,
        env,
        network,
        generation_num,
        cpu_num,
        eval_ep_num,
        log=False,
    ):
        super().__init__(env, network, cpu_num)
        self.network.init_weights(0, 1e-7)
        self.offspring_strategy = offspring_strategy
        self.generation_num = generation_num
        self.eval_ep_num = eval_ep_num
        self.ep5_rewards = deque(maxlen=5)
        self.log = log

        # create log directory
        now = datetime.now()
        curr_time = now.strftime("%Y%m%d%H%M%S")
        dir_lst = []
        self.save_dir = f"logs/{self.env.name}/{curr_time}"
        dir_lst.append(self.save_dir)
        dir_lst.append(self.save_dir + "/saved_models/")
        for _dir in dir_lst: os.makedirs(_dir)

        if log:
            wandb_cfg = self.offspring_strategy.get_wandb_cfg()
            wandb.init(project=self.env.name, config=wandb_cfg)

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
        for _ in range(self.generation_num):
            start_time = time.time()
            ep_num += 1

            # create an actor by the number of cores
            p = mp.Pool(self.cpu_num)
            arguments = [(self.env, offsprings, worker_id, self.eval_ep_num) for worker_id in range(self.cpu_num)]

            # start ollout actors
            rollout_start_time = time.time()

            # rollout
            outputs = p.map(RolloutWorker, arguments)
            # concat output lists to single list
            results = []
            for li in outputs:
                results[0:0] = li  # fast way to concatenate lists
            p.close()
            rollout_consumed_time = time.time() - rollout_start_time

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
            if self.log:
                self.ep5_rewards.append(best_reward)
                ep5_mean_reward = sum(self.ep5_rewards) / len(self.ep5_rewards)
                wandb.log(
                    {"ep5_mean_reward": ep5_mean_reward, "curr_sigma": curr_sigma}
                )
            
            elite_group = self.offspring_strategy.get_elite_model()
            save_pth = self.save_dir + "/saved_models" + f"/ep_{ep_num}/"
            os.makedirs(save_pth)
            for k, model in elite_group.items():
                torch.save(model.state_dict(), save_pth + f"{k}.pt")

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
            rollout_worker = RolloutWorker(self.env, offsprings, 0)
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

# offspring_id, worker_id, eval_ep_num=10
def RolloutWorker(arguments):
    env, offspring_id, worker_id, eval_ep_num = arguments
    rewards = []
    for i, models in enumerate(offspring_id[worker_id]):
        total_reward = 0
        for _ in range(eval_ep_num):
            states = env.reset()
            hidden_states = {}
            done = False
            for k, model in models.items():
                model.reset()
            while not done:
                actions = {}
                with torch.no_grad():
                    for k, model in models.items():
                        s = torch.from_numpy(
                            states[k]["state"][np.newaxis, ...]
                        ).float()
                        actions[k] = model(s)
                states, r, done, info = env.step(actions)
                # env.render()
                total_reward += r
        rewards.append([(worker_id, i), total_reward / eval_ep_num])
    return rewards
