from copy import deepcopy
import multiprocessing as mp
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class Agent(nn.Module):
    def __init__(self,
        D_in = 2,
        D_out = 4,
        D_hidden = [80, 80],
        ):
        super(Agent, self).__init__()
        self.layers = []
        in_size = D_in
        D_hidden.append(D_out)
        for i, out_size in enumerate(D_hidden):
            tmp_layer = nn.Linear(in_size, out_size)
            with torch.no_grad():
                tmp_layer.weight = nn.init.normal_(tmp_layer.weight)
            in_size = out_size
            self.layers.append(tmp_layer)

    def forward(self,x):
        for hidden_layer in self.layers:
            x = hidden_layer(x)
        return x

class CartPoleES:
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
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.num_process = num_process

        self.last_max_reward = -1e6
        self.elite_num = int(self.population_size * self.elite_ratio)


        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        init_agent = Agent(self.obs_dim, self.action_dim, [20])
        self.elite_param = []
        self.sigma = []
        for layer in init_agent.layers:
            self.elite_param.append(layer.weight.data.numpy())
            self.sigma.append(np.ones(layer.weight.data.shape).astype(np.float32) * 0.5)
        self.elite_param = np.array(self.elite_param)

    def gen_offspring(self, parent, sigma, offspring_num):
        offspring = []
        for i in range(offspring_num):
            tmp_agent = Agent(self.obs_dim, self.action_dim, [20])
            for j in range(len(tmp_agent.layers)):
                with torch.no_grad():
                    if isinstance(sigma, int):
                        weight = np.random.normal(parent[j], sigma)
                    else:
                        weight = np.random.normal(parent[j], sigma[j])
                    weight = torch.from_numpy(weight).float()
                    tmp_agent.layers[j].weight.data = weight
            offspring.append(tmp_agent)
        return offspring
    

    def interact(self, arguments):
        elite = arguments[0]
        sigma = arguments[1]
        population_per_process = arguments[2]
        population = self.gen_offspring(elite, sigma, population_per_process)
        population.append(self.gen_offspring(elite, 0, 1)[0])
        # 환경과 상호작용
        result = []
        with torch.no_grad():
            for agent in population:
                episode_reward = 0
                for _ in range(10):
                    s = self.env.reset()
                    for i in range(300):
                        a = agent(torch.Tensor(s).float()).argmax()
                        s, r, d, _ = self.env.step(a.numpy())
                        episode_reward += r
                        if d:
                            break

                result.append([agent, episode_reward])
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
            population_params = np.array(population_params)

            # check max reward
            if rewards[0][1] > self.last_max_reward:
                self.last_max_reward = rewards[0][1]

                # break if max_reward reached to target_return
                if self.last_max_reward >= 1400:
                    self.elite_param = population_params[0]
                    break

            # save elite_param by calculating the mean value of ranked parameters
            self.elite_param = np.mean(population_params[:self.elite_num], axis=0)

            # print mean reward of ranked agents
            elite_mean_reward = sum([x[1] for x in rewards[:self.elite_num]]) / self.elite_num
            print("iter : %d\telite_mean_reward: %f" % (i, elite_mean_reward))
            
            # calculate sigma
            tmp = []
            for param in population_params:
                tmp.append(np.power(param - self.elite_param, 2))
            tmp = np.array(tmp)
            tmp = np.mean(tmp, axis=0)
            tmp = [np.sqrt(x) for x in tmp]
            tmp = np.array(tmp)
            self.sigma = tmp
            #sigma = np.sqrt(np.mean(np.power(self.elite_param - population_param, 2), axis=0))

        #test
        agent = self.gen_offspring(self.elite_param, 0, 1)[0]
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
        print("process : %d, time: "%(self.num_process), time.time() - start)
