from copy import deepcopy

import numpy as np
import torch

from .abstracts import BaseOffspringStrategy


class simple_genetic(BaseOffspringStrategy):
    def __init__(self, init_sigma, sigma_decay, elite_num, offspring_num):
        super(simple_genetic, self).__init__()
        self.elite_num = elite_num
        self.offspring_num = offspring_num
        self.elite_models = []
        self.init_sigma = init_sigma
        self.sigma_decay = sigma_decay

        self.curr_sigma = self.init_sigma
        self.elite_num = elite_num

    @staticmethod
    def _gen_mutation(agent_ids, elite_model: dict, sigma: object, offspring_num: int):
        offsprings_group = []
        for _ in range(offspring_num):
            agent_group = {}
            for agent_id in agent_ids:
                tmp_agent = deepcopy(elite_model)
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
            # add elite
            agent_group = {}
            for agent_id in self.agent_ids:
                agent_group[agent_id] = p
            offspring_array.append(agent_group)

            # add elite offsprings
            offspring_array[0:0] = self._gen_mutation(
                self.agent_ids,
                p,
                sigma=self.curr_sigma,
                offspring_num=(self.offspring_num // self.elite_num) - 1,
            )
        return offspring_array

    def get_elite_model(self):
        return self.elite_models[0]

    def init_offspring(self, network, agent_ids):
        self.agent_ids = agent_ids
        network.init_weights(0, 1e-7)
        self.elite_models = [network for i in range(self.elite_num)]
        return self._gen_offsprings()

    def evaluate(self, result, offsprings):
        elite_ids = np.flip(np.argsort(np.array(result)))[: self.elite_num]
        best_reward = max(result)
        self.elite_models = []
        for elite_id in elite_ids:
            self.elite_models.append(offsprings[elite_id][self.agent_ids[0]])
        offsprings = self._gen_offsprings()
        self.curr_sigma *= self.sigma_decay
        return offsprings, best_reward, self.curr_sigma

    def get_wandb_cfg(self):
        wandb_cfg = dict(
            init_sigma=self.init_sigma,
            sigma_decay=self.sigma_decay,
            elite_num=self.elite_num,
            offspring_num=self.offspring_num,
        )
        return wandb_cfg


class simple_evolution(BaseOffspringStrategy):
    def __init__(self, init_sigma, sigma_decay, elite_num, offspring_num):
        super(simple_evolution, self).__init__()
        self.elite_num = elite_num
        self.offspring_num = offspring_num
        self.init_sigma = init_sigma
        self.elite_models = []
        self.elite_num = elite_num
        self.sigma_decay = sigma_decay

        self.model_mu = None
        self.model_sigma = None

    @staticmethod
    def _gen_mutation(agent_ids, mu, sigma, offspring_num):
        offsprings_group = []
        for _ in range(offspring_num):
            tmp_agent = deepcopy(mu)
            for param, sigma_param in zip(tmp_agent.parameters(), sigma.parameters()):
                with torch.no_grad():
                    param.data = torch.normal(mean=param.data, std=sigma_param.data)
            agent_group = {}
            for agent_id in agent_ids:
                agent_group[agent_id] = tmp_agent
            offsprings_group.append(agent_group)
        return offsprings_group

    def _gen_offsprings(self):
        offspring_array = []
        agent_group = {}
        for agent_id in self.agent_ids:
            agent_group[agent_id] = self.model_mu
        offspring_array.append(agent_group)
        agent_group = {}
        for agent_id in self.agent_ids:
            agent_group[agent_id] = self.elite_models[0]
        offspring_array.append(agent_group)
        offspring_array[0:0] = self._gen_mutation(
            self.agent_ids,
            self.model_mu,
            sigma=self.model_sigma,
            offspring_num=self.offspring_num - 1,
        )
        return offspring_array

    def get_elite_model(self):
        return self.elite_models[0]

    def init_offspring(self, network, agent_ids):
        self.agent_ids = agent_ids
        network.init_weights(0, 1e-7)
        self.elite_models = [network for i in range(self.elite_num)]
        self.model_mu = self.elite_models[0]
        self.model_sigma = self.model_mu
        for param in self.model_sigma.parameters():
            with torch.no_grad():
                param.data = torch.ones(param.data.size()) * self.init_sigma
        return self._gen_offsprings()

    def evaluate(self, result, offsprings):
        elite_ids = np.flip(np.argsort(np.array(result)))[: self.elite_num]
        best_reward = max(result)
        self.elite_models = []
        for elite_id in elite_ids:
            self.elite_models.append(offsprings[elite_id][self.agent_ids[0]])

        sigma_mean = []
        # simply decay sigma
        for var_param in self.model_sigma.parameters():
            with torch.no_grad():
                var_param.data = var_param.data * self.sigma_decay
                sigma_mean.append(
                    torch.sum(var_param.data) / (var_param.data.view(-1, 1).shape[0])
                )
        curr_sigma = sum(sigma_mean) / len(sigma_mean)

        # get new mu
        new_model_mu = self.elite_models[0]
        for elite in self.elite_models[1:]:
            for param, param2 in zip(new_model_mu.parameters(), elite.parameters()):
                with torch.no_grad():
                    param.data = param.data + param2.data
        # get mean weight
        for param in new_model_mu.parameters():
            with torch.no_grad():
                param.data = param.data / self.elite_num
        self.model_mu = new_model_mu

        offsprings = self._gen_offsprings()
        return offsprings, best_reward, curr_sigma

    def get_wandb_cfg(self):
        wandb_cfg = dict(
            init_sigma=self.init_sigma,
            elite_num=self.elite_num,
            offspring_num=self.offspring_num,
        )
        return wandb_cfg
