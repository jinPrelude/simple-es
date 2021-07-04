from copy import deepcopy

import numpy as np
import torch

from .abstracts import BaseOffspringStrategy
from .utils import wrap_agentid


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
    def _gen_offsprings(
        agent_ids: list,
        elite_models: list,
        elite_num: int,
        offspring_num: int,
        sigma: float,
    ):
        """Return offsprings based on current elite models.

        Parameters
        ----------
        agent_ids: list[str, ...]
        elite_models: list[torch.nn.Module, ...]
        elite_num: int
            number of the elite models.
        offspring_num: int
            number of offsprings should be made
        sigma: float
            sigma for model perturbation.

        Returns
        -------
        offsprings_group: list[dict, ...]
        """
        offspring_group = []
        for p in elite_models:
            # add elite
            offspring_group.append(wrap_agentid(agent_ids, p))
            # add elite offsprings
            for _ in range((offspring_num // elite_num) - 1):
                perturbed_network = deepcopy(p)
                for param in perturbed_network.parameters():
                    with torch.no_grad():
                        noise = torch.normal(0, sigma, size=param.size())
                        param.add_(noise)
                offspring_group.append(wrap_agentid(agent_ids, perturbed_network))
        return offspring_group

    def get_elite_model(self):
        return self.elite_models[0]

    def init_offspring(self, network: torch.nn.Module, agent_ids: list):
        """Get network and agent ids, and return initialized offsprings.

        Parameters
        ----------
        network : torch.nn.Module
            network of the agent.
        agent_ids : list[str, ...]

        Returns
        -------
        offsprings: list
            Initialized offsprings.
        """

        self.agent_ids = agent_ids
        network.init_weights(0, 1e-7)
        self.elite_models = [network for _ in range(self.elite_num)]
        offspring_group = self._gen_offsprings(
            self.agent_ids,
            self.elite_models,
            self.elite_num,
            self.offspring_num,
            self.curr_sigma,
        )
        return offspring_group

    def evaluate(self, rewards: list, offsprings: list):
        """Get rewards and offspring models, evaluate and update, and return new offsprings.

        Parameters
        ----------
        rewards : list[float, ...]
            Rewards received by offsprings
        offsprings : list[dict, ...]
            Model of the offsprings.

        Returns
        -------
        offspring_group: list
            New offsprings from updated models.
        best_reward: float
            Best rewards one of the offspring got.
        curr_sigma: float
            Current decayed sigma.
        """

        elite_ids = np.flip(np.argsort(np.array(rewards)))[: self.elite_num]
        best_reward = max(rewards)
        self.elite_models = []
        for elite_id in elite_ids:
            self.elite_models.append(offsprings[elite_id][self.agent_ids[0]])
        offspring_group = self._gen_offsprings(
            self.agent_ids,
            self.elite_models,
            self.elite_num,
            self.offspring_num,
            self.curr_sigma,
        )
        self.curr_sigma *= self.sigma_decay
        return offspring_group, best_reward, self.curr_sigma

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

        self.mu_model = None
        self.sigma_model = None

    @staticmethod
    def _gen_offsprings(agent_ids, elite_models, mu_model, sigma_model, offspring_num):
        """Return offsprings based on current elite models.

        Parameters
        ----------
        agent_ids: list[str, ...]
        elite_models: list[torch.nn.Module, ...]
        offspring_num: int
            number of offsprings should be made
        mu_model: torch.nn.Module
        sigma_model: torch.nn.Module

        Returns
        -------
        offsprings_group: list[dict, ...]
        """

        offspring_group = []
        offspring_group.append(wrap_agentid(agent_ids, mu_model))
        offspring_group.append(wrap_agentid(agent_ids, elite_models[0]))

        for _ in range(offspring_num - 1):
            preturbed_agent = deepcopy(mu_model)
            for param, sigma_param in zip(
                preturbed_agent.parameters(), sigma_model.parameters()
            ):
                with torch.no_grad():
                    param.data = torch.normal(mean=param.data, std=sigma_param.data)
            offspring_group.append(wrap_agentid(agent_ids, preturbed_agent))

        return offspring_group

    def get_elite_model(self):
        return self.elite_models[0]

    def init_offspring(self, network: torch.nn.Module, agent_ids: list):
        """Get network and agent ids, and return initialized offsprings.

        Parameters
        ----------
        network : torch.nn.Module
            network of the agent.
        agent_ids : list[str, ...]

        Returns
        -------
        offsprings: list
            Initialized offsprings.
        """
        self.agent_ids = agent_ids
        network.init_weights(0, 1e-7)
        self.elite_models = [network for i in range(self.elite_num)]
        self.mu_model = self.elite_models[0]
        self.sigma_model = self.mu_model
        for param in self.sigma_model.parameters():
            with torch.no_grad():
                param.data = torch.ones(param.data.size()) * self.init_sigma
        # agent_ids, elite_models, mu_model, sigma_model, offspring_num
        offspring_group = self._gen_offsprings(
            self.agent_ids,
            self.elite_models,
            self.mu_model,
            self.sigma_model,
            self.offspring_num,
        )
        return offspring_group

    def evaluate(self, rewards: list, offsprings: list):
        """Get rewards and offspring models, evaluate and update the elite
        model and return new offsprings.

        Parameters
        ----------
        rewards : list[float, ...]
            Rewards received by offsprings
        offsprings : list[dict, ...]
            Model of the offsprings.

        Returns
        -------
        offspring_group: list
            New offsprings from updated models.
        best_reward: float
            Best rewards one of the offspring got.
        curr_sigma: float
            Current decayed sigma.
        """

        elite_ids = np.flip(np.argsort(np.array(rewards)))[: self.elite_num]
        best_reward = max(rewards)
        self.elite_models = []
        for elite_id in elite_ids:
            self.elite_models.append(offsprings[elite_id][self.agent_ids[0]])

        sigma_mean = []
        # simply decay sigma
        for var_param in self.sigma_model.parameters():
            with torch.no_grad():
                var_param.data = var_param.data * self.sigma_decay
                sigma_mean.append(
                    torch.sum(var_param.data) / (var_param.data.view(-1, 1).shape[0])
                )
        curr_sigma = sum(sigma_mean) / len(sigma_mean)

        # get new mu
        new_mu_model = self.elite_models[0]
        for elite in self.elite_models[1:]:
            for param, param2 in zip(new_mu_model.parameters(), elite.parameters()):
                with torch.no_grad():
                    param.data = param.data + param2.data
        # get mean weight
        for param in new_mu_model.parameters():
            with torch.no_grad():
                param.data = param.data / self.elite_num
        self.mu_model = new_mu_model

        offspring_group = self._gen_offsprings(
            self.agent_ids,
            self.elite_models,
            self.mu_model,
            self.sigma_model,
            self.offspring_num,
        )
        return offspring_group, best_reward, curr_sigma

    def get_wandb_cfg(self):
        wandb_cfg = dict(
            init_sigma=self.init_sigma,
            elite_num=self.elite_num,
            offspring_num=self.offspring_num,
        )
        return wandb_cfg
