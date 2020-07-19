import multiprocessing as mp
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class BasicEs:
    def __init__(
        self,
        env,
        epoch=1000,
        mu_init=10,
        sigma_init=0.5,
        population_size=100,
        elite_ratio=0.1,
        num_process=2,
    ):

        self.env = env
        self.epoch = epoch
        self.mu = mu_init
        self.sigma = sigma_init
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.num_process = num_process

        self.last_max_reward = -1e6
        self.elite_num = int(self.population_size * self.elite_ratio)
        self.elite_param = np.ones((2)) * self.mu

    def interact(self, arguments):
        elite = arguments[0]
        sigma = arguments[1]
        population_per_process = arguments[2]
        population = []
        for _ in range(population_per_process):
            population.append(np.random.normal(elite, scale=sigma))
        # 환경과 상호작용
        result = []
        for agent in population:
            self.env.reset()
            while True:
                _, reward, d, _ = self.env.step(agent)
                if d:
                    break
            result.append([agent, reward])
        return result
    """
    def _slice(self, lst, n):
        # 리스트를 n개만큼 자름
        result = []
        size = int(len(lst) / n)
        for i in range(0, n):
            result.append(lst[i * size : (i + 1) * size])
        if size * n < len(lst):
            j = size * n
            k = 0
            # assign extra data
            while j < len(lst):
                result[k].append(lst[j])
                j += 1
                k += 1
                if k >= len(result):
                    k = 0
        return result
    """
    def run(self):
        import time
        start = time.time()
        p = mp.Pool(self.num_process)
        population_per_process = int(self.population_size / self.num_process)
        for i in range(self.epoch):
            arguments = [[self.elite_param, self.sigma, population_per_process] for _ in range(self.num_process)]
            outputs = p.map(self.interact, arguments)
            rewards = []
            for li in outputs:
                rewards[0:0] = li  # fast way to concatenate lists
            rewards = sorted(rewards, key=lambda l: l[1], reverse=True)

            # update
            population_param = np.array([x[0] for x in rewards])
            if rewards[0][1] > self.last_max_reward:
                self.last_max_reward = rewards[0][1]
                self.elite_param = np.mean(population_param[:self.elite_num], axis=0)
                print("iter : %d\tmax reward: %f" % (i, self.last_max_reward))
                # 부산
            tmp = []
            for param in population_param:
                tmp.append(np.power(param - self.elite_param, 2))
            tmp = np.array(tmp)
            tmp = np.mean(tmp, axis=0)
            tmp = np.sqrt(tmp)
            #sigma = np.sqrt(np.mean(np.power(self.elite_param - population_param, 2), axis=0))
            self.sigma = tmp
            print(self.sigma)
        print("best : ", self.elite_param)
        print("process : %d, time: "%(self.num_process), time.time() - start)
