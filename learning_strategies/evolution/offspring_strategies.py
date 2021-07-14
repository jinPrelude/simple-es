from copy import deepcopy

import numpy as np
import torch

from .abstracts import BaseOffspringStrategy
from .utils import wrap_agentid
from learning_strategies.optimizers import Adam


class simple_genetic(BaseOffspringStrategy):
    def __init__(self, init_sigma, sigma_decay, elite_num, offspring_num):
        super(simple_genetic, self).__init__()
        self.elite_num = elite_num
        self.offspring_num = offspring_num
        self.elite_models = []
        self.init_sigma = init_sigma
        self.sigma_decay = sigma_decay
        self.offsprings = []

        self.curr_sigma = self.init_sigma

    def _gen_offsprings(
        self,
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
        self.offsprings = []
        for p in elite_models:
            # add elite
            self.offsprings.append(p)
            # add elite offsprings
            for _ in range((offspring_num // elite_num) - 1):
                perturbed_network = deepcopy(p)
                perturbed_network_param_list = perturbed_network.get_param_list()
                for param in perturbed_network_param_list:
                    noise = np.random.normal(0, sigma, size=param.shape)
                    param += noise
                perturbed_network.apply_param(perturbed_network_param_list)
                self.offsprings.append(perturbed_network)
        offspring_group = [wrap_agentid(agent_ids, model) for model in self.offsprings]
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
        network.zero_init()
        self.elite_models = [network for _ in range(self.elite_num)]
        offspring_group = self._gen_offsprings(
            self.agent_ids,
            self.elite_models,
            self.elite_num,
            self.offspring_num,
            self.curr_sigma,
        )
        return offspring_group

    def evaluate(self, rewards: list):
        """Get rewards and offspring models, evaluate and update, and return new offsprings.

        Parameters
        ----------
        rewards : list[float, ...]
            Rewards received by offsprings

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
            self.elite_models.append(self.offsprings[elite_id])
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
        self.sigma_decay = sigma_decay
        self.curr_sigma = self.init_sigma
        self.offsprings = []

        self.mu_model = None

    def _gen_offsprings(self, agent_ids, elite_models, mu_model, sigma, offspring_num):
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
        self.offsprings = []
        self.offsprings.append(mu_model)
        self.offsprings.append(elite_models[0])

        for _ in range(offspring_num - 1):
            preturbed_net = deepcopy(mu_model)
            perturbed_net_param_list = preturbed_net.get_param_list()
            for param in perturbed_net_param_list:
                epsilon = np.random.normal(0, sigma, size=param.shape)
                param += epsilon
            preturbed_net.apply_param(perturbed_net_param_list)
            self.offsprings.append(preturbed_net)

        offspring_group = [wrap_agentid(agent_ids, model) for model in self.offsprings]

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
        network.zero_init()
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

    def evaluate(self, rewards: list):
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
            self.elite_models.append(self.offsprings[elite_id])

        # get new mu
        new_mu_param_list = self.elite_models[0].get_param_list()
        for elite in self.elite_models[1:]:
            elite_param_list = elite.get_param_list()
            for mu_param, elite_param in zip(new_mu_param_list, elite_param_list):
                mu_param += elite_param
        # get mean weight
        for param in new_mu_param_list:
            param /= self.elite_num

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


class openai_es(BaseOffspringStrategy):
    def __init__(self, init_sigma, sigma_decay, learning_rate, offspring_num):
        super(openai_es, self).__init__()
        self.offspring_num = offspring_num
        self.init_sigma = init_sigma
        self.sigma_decay = sigma_decay
        self.learning_rate = learning_rate
        self.curr_sigma = self.init_sigma
        self.epsilons = []

        self.mu_model = None
        self.elite_model = None
        self.optimizer = None

    def _gen_offsprings(self, agent_ids, mu_model, sigma, offspring_num):
        """Return offsprings based on current elite models.

        Parameters
        ----------
        agent_ids: list[str, ...]
        mu_model: torch.nn.Module
        sigma: float
        offspring_num: int
            number of offsprings should be made

        Returns
        -------
        offsprings_group: list[dict, ...]
        """
        self.epsilons = []
        offspring_group = []

        # epsilon of the pure mo_model is zero.
        zero_net = deepcopy(self.mu_model)
        zero_net_param_list = zero_net.get_param_list()
        for param in zero_net_param_list:
            param = np.zeros(param.shape)
        zero_net.apply_param(zero_net_param_list)
        self.epsilons.append(zero_net)

        offspring_group.append(wrap_agentid(agent_ids, self.mu_model))

        for _ in range(offspring_num - 1):
            preturbed_net = deepcopy(mu_model)
            epsilon_net = deepcopy(mu_model)
            perturbed_net_param_list = preturbed_net.get_param_list()
            eps_net_param_list = deepcopy(zero_net_param_list)
            for eps_param, perturb_param in zip(
                eps_net_param_list, perturbed_net_param_list
            ):
                epsilon = np.random.normal(size=perturb_param.shape)
                eps_param += epsilon
                perturb_param += epsilon * sigma
            preturbed_net.apply_param(perturbed_net_param_list)
            offspring_group.append(wrap_agentid(agent_ids, preturbed_net))
            epsilon_net.apply_param(eps_net_param_list)
            self.epsilons.append(epsilon_net)

        return offspring_group

    def get_elite_model(self):
        return self.mu_model

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
        network.zero_init()
        self.elite_model = network
        self.mu_model = network
        self.optimizer = Adam(self.mu_model, self.learning_rate)
        # agent_ids, elite_models, mu_model, sigma_model, offspring_num
        offspring_group = self._gen_offsprings(
            self.agent_ids,
            self.mu_model,
            self.curr_sigma,
            self.offspring_num,
        )
        return offspring_group

    def evaluate(self, rewards: list):
        """Get rewards and offspring models, evaluate and update the elite
        model and return new offsprings.

        Parameters
        ----------
        rewards : list[float, ...]
            Rewards received by offsprings

        Returns
        -------
        offspring_group: list
            New offsprings from updated models.
        best_reward: float
            Best rewards one of the offspring got.
        curr_sigma: float
            Current decayed sigma.
        """

        offspring_rank_id = np.flip(np.argsort(np.array(rewards)))
        best_reward = max(rewards)

        # # reconstruct elite model using epsilon and sigma
        # self.elite_model = deepcopy(self.mu_model)
        # elite_param_list = self.elite_model.get_param_list()
        # elite_epsilon = self.epsilons[offspring_rank_id[0]]
        # eps_param_list = elite_epsilon.get_param_list()
        # for elite_param, eps_param in zip(elite_param_list, eps_param_list):
        #     elite_param += eps_param * self.curr_sigma
        # self.elite_model.apply_param(elite_param_list)

        reward_array = np.zeros(len(rewards))
        for idx in reversed(range(len(rewards))):
            reward_array[offspring_rank_id[idx]] = (
                (len(rewards) - 1 - idx) / (len(rewards) - 1)
            ) - 0.5
        r_std = reward_array.std()
        reward_array = (reward_array - reward_array.mean()) / r_std

        # get new mu
        grad = deepcopy(self.mu_model)
        grad_param_list = grad.get_param_list()
        for grad_param in grad_param_list:
            grad_param *= 0

        update_factor = self.learning_rate / (len(self.epsilons) * self.curr_sigma)
        # multiply negative num to make minimize problem for optimizer
        update_factor *= -1.0
        for offs_idx, offs in enumerate(self.epsilons):
            offs_param_list = offs.get_param_list()
            for grad_param, offs_param in zip(grad_param_list, offs_param_list):
                grad_param += offs_param * reward_array[offs_idx]
        for grad_param in grad_param_list:
            grad_param *= update_factor

        self.optimizer.update(grad_param_list)

        self.curr_sigma *= self.sigma_decay
        offspring_group = self._gen_offsprings(
            self.agent_ids,
            self.mu_model,
            self.curr_sigma,
            self.offspring_num,
        )
        return offspring_group, best_reward, self.curr_sigma

    def get_wandb_cfg(self):
        wandb_cfg = dict(
            init_sigma=self.init_sigma,
            sigma_decay=self.sigma_decay,
            learning_rate=self.learning_rate,
            offspring_num=self.offspring_num,
        )
        return wandb_cfg
