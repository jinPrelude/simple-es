from copy import deepcopy

import torch


class simple_gaussian_offspring:
    def __init__(self, init_sigma, sigma_decay, elite_ratio, group_num):
        self.init_sigma = init_sigma
        self.sigma_decay = sigma_decay
        self.elite_ratio = elite_ratio
        self.group_num = group_num
        self.elite_models = []

        self.curr_sigma = self.init_sigma
        self.elite_num = max(1, int(group_num * elite_ratio))

    @staticmethod
    def _gen_mutation(group: dict, sigma: object, group_num: int):
        offsprings_group = []
        for _ in range(group_num):
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
                group_num=(self.group_num // self.elite_num) - 1,
            )
        return offspring_array

    def get_elite_models(self):
        return self.elite_models

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