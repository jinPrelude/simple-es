import logging
import random
import yaml
import argparse
import numpy as np
import torch
import builder
from learning_strategies.evolution.loop import ESLoop


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-config', type=str, default="conf/simple_spread/env.yaml")
    parser.add_argument('--network-config', type=str, default="conf/simple_spread/gym_model.yaml")
    parser.add_argument('--strategy-config', type=str, default="conf/simple_spread/simple_genetic.yaml")
    parser.add_argument('--process-num', type=int, default=4)
    parser.add_argument('--generation-num', type=int, default=300)
    parser.add_argument('--eval-ep-num', type=int, default=3)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--save-model-period', type=int, default=10)
    args = parser.parse_args()

    with open(args.env_config) as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
    with open(args.network_config) as f:
        network_config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
    with open(args.strategy_config) as f:
        strategy_config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    env = builder.build_env(env_config)
    network = builder.build_network(network_config)
    loop = builder.build_strategy(strategy_config, env, network, args.generation_num, args.process_num, args.eval_ep_num, args.log)
    loop.run()


if __name__ == "__main__":
    main()
