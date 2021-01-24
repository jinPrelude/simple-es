import os

import numpy as np
import ray
import torch


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
