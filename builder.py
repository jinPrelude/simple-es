from envs.gym_wrapper import *
from envs.pettingzoo_wrapper import *
from networks.neural_network import *
from learning_strategies.evolution.offspring_strategies import *
from learning_strategies.evolution.loop import *

def build_env(config):
    if config['name'] in ['simple_spread', 'waterworld', 'multiwalker']:
        return PettingzooWrapper(config['name'], config['max_step'])
    else:
        return GymWrapper(config['name'], config['max_step'])

def build_network(config):
    if config['name'] == "gym_model":
        return GymEnvModel(config['num_state'], config['num_action'],
                            config['discrete_action'], config['gru'])

def build_strategy(config, env, network, gen_num, process_num, eval_ep_num, log, save_model_period):
    if config['name'] == 'simple_evolution':
        strategy = simple_evolution(config['init_sigma'], config['elite_num'],
                                    config['offspring_num'], config['sigma_decay'],
                                    config['sigma_decay_method'])
        return ESLoop(strategy, env, network, gen_num, process_num, eval_ep_num, log, save_model_period)

    elif config['name'] == 'simple_genetic':
        strategy = simple_genetic(config['init_sigma'], config['sigma_decay'],
                                    config['elite_num'], config['offspring_num'])
        return ESLoop(strategy, env, network, gen_num, process_num, eval_ep_num, log, save_model_period)