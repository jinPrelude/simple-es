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
                perturbed_network_param_list = perturbed_network.get_param_list()
                for idx in range(len(perturbed_network_param_list)):
                    noise = np.random.normal(
                        0, sigma, size=perturbed_network_param_list[idx].shape
                    )
                    perturbed_network_param_list[idx] += noise
                perturbed_network.apply_param(perturbed_network_param_list)
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
        self.curr_sigma = self.init_sigma

        self.mu_model = None

    @staticmethod
    def _gen_offsprings(agent_ids, elite_models, mu_model, sigma, offspring_num):
        """Return offsprings based on current elite models.

        Parameters
        ----------
        agent_ids: list[str, ...]
        elite_models: list[torch.nn.Module, ...]
        offspring_num: int
            number of offsprings should be made
        mu_model: torch.nn.Module

        Returns
        -------
        offsprings_group: list[dict, ...]
        """

        offspring_group = []
        offspring_group.append(wrap_agentid(agent_ids, mu_model))
        offspring_group.append(wrap_agentid(agent_ids, elite_models[0]))

        for _ in range(offspring_num - 1):
            preturbed_net = deepcopy(mu_model)
            perturbed_net_param_list = preturbed_net.get_param_list()
            for idx in range(len(perturbed_net_param_list)):
                epsilon = np.random.normal(
                    0, sigma, size=perturbed_net_param_list[idx].shape
                )
                perturbed_net_param_list[idx] += epsilon
            preturbed_net.apply_param(perturbed_net_param_list)
            offspring_group.append(wrap_agentid(agent_ids, preturbed_net))

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
        # agent_ids, elite_models, mu_model, sigma_model, offspring_num
        offspring_group = self._gen_offsprings(
            self.agent_ids,
            self.elite_models,
            self.mu_model,
            self.curr_sigma,
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

        # get new mu
        new_mu_param_list = self.elite_models[0].get_param_list()
        for elite in self.elite_models[1:]:
            elite_param_list = elite.get_param_list()
            for idx in range(len(elite_param_list)):
                new_mu_param_list[idx] += elite_param_list[idx]
        # get mean weight
        for idx in range(len(new_mu_param_list)):
            new_mu_param_list[idx] /= self.elite_num

        self.mu_model.apply_param(new_mu_param_list)
        self.curr_sigma *= self.sigma_decay
        offspring_group = self._gen_offsprings(
            self.agent_ids,
            self.elite_models,
            self.mu_model,
            self.curr_sigma,
            self.offspring_num,
        )
        return offspring_group, best_reward, self.curr_sigma

    def get_wandb_cfg(self):
        wandb_cfg = dict(
            init_sigma=self.init_sigma,
            elite_num=self.elite_num,
            offspring_num=self.offspring_num,
        )
        return wandb_cfg
