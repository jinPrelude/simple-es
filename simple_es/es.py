from copy import deepcopy
import time
import random
import multiprocessing as mp
import numpy as np
import torch
import gym
from torch import nn
import torch.nn.functional as F
import wandb
import pickle
import ray
from simple_es.utils import slice_list

@ray.remote
def interact(env, population, max_episode_step, is_continuous_action) -> list:
        # 환경과 상호작용
        result = []
        with torch.no_grad():
            for agent in population:
                episode_reward = 0
                for _ in range(10):
                    s = env.reset()
                    for _ in range(max_episode_step):
                        a = agent(torch.Tensor(s).float())
                        if not is_continuous_action: a = a.argmax()
                        s, r, d, _ = env.step(a.numpy())
                        episode_reward += r
                        if d:
                            break
                episode_reward /= 10
                result.append([agent, episode_reward])
        return result
"""
@ray.remote
def gen_offspring(parent: object, sigma: object, offspring_num: int, model, obs_dim, action_dim, hidden_dim) -> list:
        offspring = []
        for _ in range(offspring_num):
            tmp_agent = model(obs_dim, action_dim, hidden_dim)
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
"""
@ray.remote
class gen_offspring(object):
    def __init__(self, model, obs_dim, action_dim, hidden_dim):
        self.model = model
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.offspring = []
    def set(self, parent: object, sigma: object, offspring_num: int):
        offspring = []
        tmp_agent = self.model(self.obs_dim, self.action_dim, deepcopy(self.hidden_dim))
        for _ in range(offspring_num):
            for j in range(len(tmp_agent.layers)):
                with torch.no_grad():
                    if isinstance(sigma, int):
                        weight = np.random.normal(parent[j], sigma)
                    else:
                        weight = np.random.normal(parent[j], sigma[j])
                    weight = torch.Tensor(weight).float()
                    tmp_agent.layers[j].weight.data = weight
            offspring.append(tmp_agent)
        self.offspring = offspring
    
    def get(self):
        return self.offspring

class ES:
    def __init__(
        self,
        model: nn.Module,
        env: gym.Env,
        is_continuous_action: bool,
        max_episode_step: int,
        seed: int,
        std_init: float = 0.5,
        std_decay: int = 0.99,
        epoch: int = 1000,
        population_size: int = 100,
        elite_ratio: int = 0.1,
        num_process: int = 2,
        target_reward: int = 300,
        wandb_log: bool = True,
    ):
        self.env = env
        self.is_continuous_action = is_continuous_action
        self.seed = seed
        self.std_decay = std_decay
        self.max_episode_step = max_episode_step
        self.model = model
        self.epoch = epoch
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.num_process = num_process
        self.target_reward = target_reward
        self.wandb_log = wandb_log

        ray.init(num_cpus=num_process)
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.env.seed(seed)
        random.seed(seed)
        if wandb_log:
            wandb.init(project=self.env.spec.id)

        self.last_max_reward = -1e6
        self.elite_num = int(self.population_size * self.elite_ratio)

        self.obs_dim = self.env.observation_space.shape[0]
        if self.is_continuous_action:
            self.action_dim = self.env.action_space.shape[0]
        else:
            self.action_dim = self.env.action_space.n

        self.offspring_generator = gen_offspring.remote(model, self.obs_dim, self.action_dim, [120, 120])

        init_agent = self.model(self.obs_dim, self.action_dim, [120, 120])
        self.mean_elite_param = []
        self.top_elite_param = []
        self.std = []
        for layer in init_agent.layers:
            self.mean_elite_param.append(layer.weight.data.numpy())
            self.top_elite_param.append(layer.weight.data.numpy())
            self.std.append(np.ones(layer.weight.data.shape).astype(np.float32) * std_init)
        self.mean_elite_param = np.array(self.mean_elite_param, dtype=object)
        self.top_elite_param = np.array(self.top_elite_param, dtype=object)

    def _test(
        self,
        agent_params: object,
        render: bool = False,
        print_log: bool = False,
        test_episode_num: int = 1,
    ) -> float:
        # test
        self.offspring_generator.set.remote(agent_params, 0, 1)
        agent = ray.get(self.offspring_generator.get.remote())[0]
        test_num = 10
        reward_sum = 0
        self.env.seed(self.seed)
        with torch.no_grad():
            for _ in range(test_num):
                episode_reward = 0
                s = self.env.reset()
                for _ in range(self.max_episode_step):
                    a = agent(torch.Tensor(s).float())
                    if not self.is_continuous_action: a = a.argmax()
                    s, r, d, _ = self.env.step(a.numpy())
                    if render:
                        self.env.render()
                    episode_reward += r
                    if d:
                        break
                reward_sum += episode_reward
                if print_log:
                    print("reward: %.3f" % episode_reward)
        return reward_sum / test_num

    def save_agent(self, agent: object):
        with open('./best_model.pt', "wb") as f:
            pickle.dump(self.top_elite_param, f)

    def run(self):
        start = time.time()
        #self.offspring_generator.set.remote(self.mean_elite_param, self.std, self.population_size - 1)
        
        for i in range(self.epoch):
    
            #arguments = [(j, offspring[j]) for j in range(self.num_process)]
            #outputs = p.map(self.interact, arguments)
            self.offspring_generator.set.remote(self.mean_elite_param, self.std, self.population_size - 1)
            offspring = ray.get(self.offspring_generator.get.remote())
            self.offspring_generator.set.remote(self.top_elite_param, 0, 1)
            offspring.append(ray.get(self.offspring_generator.get.remote())[0])
            offspring = slice_list(offspring, self.num_process)
            
            outputs = ray.get([interact.remote(self.env, offspring[j], self.max_episode_step, self.is_continuous_action) for j in range(self.num_process)])
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

            # save elite_param by calculating the mean value of ranked parameters
            self.mean_elite_param = deepcopy(np.mean(population_params[: self.elite_num], axis=0))
            # print mean reward of ranked agents
            sum_std = np.mean(np.array([np.mean(x) for x in self.std], dtype=object))

            

            # select top agent parameter
            new_rank1_test_r = self._test(population_params[0], test_episode_num=10)
            if new_rank1_test_r > self._test(self.top_elite_param, test_episode_num=10):
                self.top_elite_param = deepcopy(population_params[0])
                print("rank1 test mean_reward = %.3f" % (new_rank1_test_r))
            if new_rank1_test_r > self.target_reward:
                print("final test success.")
                break

            
            print(
                "iter : %d\telite_mean_reward: %f\tmean_std: %f"
                % (i, rewards[0][1], sum_std)
            )

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
                tmp_std.append(layer * self.std_decay)
            self.std = tmp_std
            if self.wandb_log:
                wandb.log({"reward": rewards[0][1]})
        print("process : %d, time: " % (self.num_process), time.time() - start)
        self.save_agent(self.top_elite_param)
        self._test(self.top_elite_param, render=True, print_log=True)
