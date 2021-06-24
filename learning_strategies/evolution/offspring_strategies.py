from copy import deepcopy

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
    def _gen_mutation(group: dict, sigma: object, offspring_num: int):
        offsprings_group = []
        for _ in range(offspring_num):
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
                offspring_num=(self.offspring_num // self.elite_num) - 1,
            )
        return offspring_array

    def get_elite_model(self):
        return self.elite_models[0]

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

    def get_wandb_cfg(self):
        wandb_cfg = dict(
            init_sigma=self.init_sigma,
            sigma_decay=self.sigma_decay,
            elite_num=self.elite_num,
            offspring_num=self.offspring_num,
        )
        return wandb_cfg


class simple_evolution(BaseOffspringStrategy):
    def __init__(
        self,
        init_sigma,
        elite_num,
        offspring_num,
        sigma_decay=0.995,
        sigma_decay_method="simple_decay",
    ):
        super(simple_evolution, self).__init__()
        self.elite_num = elite_num
        self.offspring_num = offspring_num
        self.init_sigma = init_sigma
        self.elite_models = []
        self.elite_num = elite_num
        self.sigma_decay = sigma_decay
        self.sigma_decay_method = sigma_decay_method

        self.model_mu = None
        self.model_sigma = None

    @staticmethod
    def _gen_mutation(group, sigma, offspring_num):
        offsprings_group = []
        for _ in range(offspring_num):
            agent_group = {}
            for agent_id, agent in group.items():
                tmp_agent = deepcopy(agent)
                for param, sigma_param in zip(
                    tmp_agent.parameters(), sigma[agent_id].parameters()
                ):
                    with torch.no_grad():
                        param.data = torch.normal(mean=param.data, std=sigma_param.data)

                agent_group[agent_id] = tmp_agent
            offsprings_group.append(agent_group)
        return offsprings_group

    def _gen_offsprings(self):
        offspring_array = []
        offspring_array.append(self.model_mu)
        offspring_array.append(self.elite_models[0])
        offspring_array[0:0] = self._gen_mutation(
            self.model_mu,
            sigma=self.model_sigma,
            offspring_num=self.offspring_num - 1,
        )
        return offspring_array

    def get_elite_model(self):
        return self.elite_models[0]

    def init_offspring(self, network, agent_ids):
        network.init_weights(0, 1e-7)
        for _ in range(self.elite_num):
            group = {}
            for agent_id in agent_ids:
                group[agent_id] = deepcopy(network)
            self.elite_models.append(group)
        self.model_mu = self.elite_models[0]
        self.model_sigma = self.model_mu
        for agent_id, agent in self.model_sigma.items():
            for param in agent.parameters():
                with torch.no_grad():
                    param.data = torch.ones(param.data.size()) * self.init_sigma
        return self._gen_offsprings()

    def evaluate(self, result, offsprings):
        results = sorted(result, key=lambda l: l[1], reverse=True)
        best_reward = results[0][1]
        elite_ids = results[: self.elite_num]
        self.elite_models = []
        for elite_id in elite_ids:
            self.elite_models.append(offsprings[elite_id[0][0]][elite_id[0][1]])

        sigma_mean = []
        if self.sigma_decay_method == "original":
            new_model_sigma = self.elite_models[0]
            # init model for save variances
            for agent_id, agent in new_model_sigma.items():
                for param in agent.parameters():
                    param.data = torch.zeros(param.data.size())
            # obtain and store variances
            for elite in self.elite_models:
                for agent_id, agent in elite.items():
                    for var_param, elite_param, mu_param in zip(
                        new_model_sigma[agent_id].parameters(),
                        agent.parameters(),
                        self.model_mu[agent_id].parameters(),
                    ):
                        with torch.no_grad():
                            var_param.data = var_param.data + torch.pow(
                                elite_param.data - mu_param.data, 2
                            )
            # divide into elite num and get the standard deviation.
            for agent_id, agent in new_model_sigma.items():
                for var_param in agent.parameters():
                    with torch.no_grad():
                        var_param.data = var_param.data / self.elite_num
                        var_param.data = torch.sqrt(var_param.data)
                        t = var_param.data.size()
                        sigma_mean.append(
                            torch.sum(var_param.data)
                            / (var_param.data.view(-1, 1).shape[0])
                        )
            self.model_sigma = new_model_sigma
        elif self.sigma_decay_method == "simple_decay":
            # simply decay sigma
            for agent_id, agent in self.model_sigma.items():
                for var_param in agent.parameters():
                    with torch.no_grad():
                        var_param.data = var_param.data * self.sigma_decay
                        sigma_mean.append(
                            torch.sum(var_param.data)
                            / (var_param.data.view(-1, 1).shape[0])
                        )
        curr_sigma = sum(sigma_mean) / len(sigma_mean)

        # get new mu
        new_model_mu = self.elite_models[0]
        for elite in self.elite_models[1:]:
            for agent_id, agent in elite.items():
                for param, param2 in zip(
                    new_model_mu[agent_id].parameters(), agent.parameters()
                ):
                    with torch.no_grad():
                        param.data = param.data + param2.data
        # get mean weight
        for agent_id, agent in new_model_mu.items():
            for param in new_model_mu[agent_id].parameters():
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
