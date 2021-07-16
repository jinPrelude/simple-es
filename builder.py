import yaml

from envs.gym_wrapper import *
from envs.pettingzoo_wrapper import *
from networks.neural_network import *
from learning_strategies.evolution.offspring_strategies import *
from learning_strategies.evolution.loop import *


def build_env(config):
    if config["name"] in ["simple_spread", "waterworld", "multiwalker"]:
        return PettingzooWrapper(config["name"], config["max_step"])
    else:
        return GymWrapper(config["name"], config["max_step"], config["pomdp"])


def build_network(config):
    if config["name"] == "gym_model":
        return GymEnvModel(
            config["num_state"],
            config["num_action"],
            config["discrete_action"],
            config["gru"],
        )
    if config["name"] == "AtariDQN":
        return AtariDQN(
            config["num_channel"],
            config["num_action"],
        )


def build_loop(config, gen_num, process_num, eval_ep_num, log, save_model_period):

    env = build_env(config["env"])
    network = build_network(config["network"])
    strategy_cfg = config["strategy"]

    if strategy_cfg["name"] == "simple_evolution":
        strategy = simple_evolution(
            strategy_cfg["init_sigma"],
            strategy_cfg["sigma_decay"],
            strategy_cfg["elite_num"],
            strategy_cfg["offspring_num"],
        )
        return ESLoop(
            config,
            strategy,
            env,
            network,
            gen_num,
            process_num,
            eval_ep_num,
            log,
            save_model_period,
        )
    elif strategy_cfg["name"] == "openai_es":
        strategy = openai_es(
            strategy_cfg["init_sigma"],
            strategy_cfg["sigma_decay"],
            strategy_cfg["learning_rate"],
            strategy_cfg["offspring_num"],
        )
        return ESLoop(
            config,
            strategy,
            env,
            network,
            gen_num,
            process_num,
            eval_ep_num,
            log,
            save_model_period,
        )
    elif strategy_cfg["name"] == "simple_genetic":
        strategy = simple_genetic(
            strategy_cfg["init_sigma"],
            strategy_cfg["sigma_decay"],
            strategy_cfg["elite_num"],
            strategy_cfg["offspring_num"],
        )
        return ESLoop(
            config,
            strategy,
            env,
            network,
            gen_num,
            process_num,
            eval_ep_num,
            log,
            save_model_period,
        )
