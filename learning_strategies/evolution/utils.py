from copy import deepcopy


def wrap_agentid(agent_ids, network):
    group = {}
    for agent_id in agent_ids:
        group[agent_id] = deepcopy(network)
    return group
