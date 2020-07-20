from copy import deepcopy
import random
import multiprocessing as mp
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import wandb


class ES:
    def __init__(
        self,
        model,
        env,
        max_episode_step,
        seed,
        epoch=1000,
        population_size=100,
        elite_ratio=0.1,
        num_process=2,
        target_reward=300,
    ):
        self.env = env
        self.max_episode_step = max_episode_step
        self.model = model
        self.epoch = epoch
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.num_process = num_process
        self.target_reward = target_reward

        np.random.seed(seed)
        torch.manual_seed(seed)
        self.env.seed(seed)
        random.seed(seed)

        wandb.init(project=self.env.spec.id)

        self.last_max_reward = -1e6
        self.elite_num = int(self.population_size * self.elite_ratio)

        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        init_agent = self.model(self.obs_dim, self.action_dim, [80, 80])
        self.mean_elite_param = []
        self.top_elite_param = []
        self.std = []
        for layer in init_agent.layers:
            self.mean_elite_param.append(layer.weight.data.numpy())
            self.top_elite_param.append(layer.weight.data.numpy())
            self.std.append(np.ones(layer.weight.data.shape).astype(np.float32) * 0.5)
        self.mean_elite_param = np.array(self.mean_elite_param, dtype=object)
        self.top_elite_param = np.array(self.top_elite_param, dtype=object)

    def gen_offspring(self, parent, sigma, offspring_num):
        offspring = []
        for i in range(offspring_num):
            tmp_agent = self.model(self.obs_dim, self.action_dim, [80, 80])
            for j in range(len(tmp_agent.layers)):
                with torch.no_grad():
                    if isinstance(sigma, int):
                        weight = np.random.normal(parent[j], sigma)
                    else:
                        weight = np.random.normal(parent[j], sigma[j])
                    weight = torch.Tensor(weight).float()
                    tmp_agent.layers[j].weight.data = weight
            offspring.append(tmp_agent)
        return offspring

    def interact(self, arguments):
        worker_id = arguments[0]
        population_per_process = arguments[1]

        if worker_id == 0:
            population = self.gen_offspring(
                self.mean_elite_param, self.std, population_per_process - 1
            )
            population.append(self.gen_offspring(self.top_elite_param, 0, 1)[0])
        else:
            population = self.gen_offspring(
                self.mean_elite_param, self.std, population_per_process
            )
        # population = self.gen_offspring(self.mean_elite_param, self.std, population_per_process)
        # 환경과 상호작용
        result = []
        with torch.no_grad():
            for agent in population:
                episode_reward = 0
                for _ in range(5):
                    s = self.env.reset()
                    for _ in range(self.max_episode_step):
                        a = agent(torch.Tensor(s).float()).argmax()
                        s, r, d, _ = self.env.step(a.numpy())
                        episode_reward += r
                        if d:
                            break
                episode_reward /= 5
                result.append([agent, episode_reward])
        return result

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

    def run(self):
        import time

        start = time.time()
        p = mp.Pool(self.num_process)
        population_per_process = int(self.population_size / self.num_process)
        for i in range(self.epoch):
            arguments = [[j, population_per_process] for j in range(self.num_process)]
            outputs = p.map(self.interact, arguments)

            # update

            # concat output lists to single list
            rewards = []
            for li in outputs:
                rewards[0:0] = li  # fast way to concatenate lists
            rewards = sorted(rewards, key=lambda l: l[1], reverse=True)

            # convert parameters to numpy and append to list
            population_params = []
            for x in rewards:
                agent = x[0]
                population_params.append([x.weight.data.numpy() for x in agent.layers])
            population_params = np.array(population_params, dtype=object)

            # check max reward
            if rewards[0][1] >= self.last_max_reward:
                self.last_max_reward = rewards[0][1]
                # break if max_reward reached to target_return
                if self.last_max_reward >= self.target_reward:
                    self.mean_elite_param = population_params[0]
                    break

            self.top_elite_param = population_params[0]
            # save elite_param by calculating the mean value of ranked parameters
            self.mean_elite_param = np.mean(population_params[: self.elite_num], axis=0)
            # print mean reward of ranked agents
            sum_std = [np.mean(x) for x in self.std]
            print("iter : %d\telite_mean_reward: %f\tmean_std: %f" % (i, rewards[0][1], np.sum(sum_std)))

            # calculate sigma

            # 정석
            # deviations = []
            # for param in population_params:
            #     deviations.append(np.power(param - self.mean_elite_param, 2))
            # variance = np.mean(np.array(deviations), axis=0)
            # std = [np.sqrt(x) for x in variance]
            # self.std = np.array(std, dtype=object)

            # 그냥 decay
            tmp_std = []
            for layer in self.std:
                tmp_std.append(layer * 0.99)
            self.std = tmp_std

            wandb.log({"reward": rewards[0][1]})

        # test
        agent = self.gen_offspring(self.mean_elite_param, 0, 1)[0]
        for _ in range(100):
            episode_reward = 0
            s = self.env.reset()
            for _ in range(300):
                a = agent(torch.Tensor(s).float()).argmax()
                s, r, d, _ = self.env.step(a.numpy())
                self.env.render()
                episode_reward += r
                if d:
                    break
            print(episode_reward)
        print("process : %d, time: " % (self.num_process), time.time() - start)
