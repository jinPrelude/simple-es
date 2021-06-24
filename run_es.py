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
    parser.add_argument('--config', type=str, default="conf/simplespread.yaml")
    parser.add_argument('--process-num', type=int, default=12)
    parser.add_argument('--generation-num', type=int, default=300)
    parser.add_argument('--eval-ep-num', type=int, default=3)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--save-model-period', type=int, default=10)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    env = builder.build_env(config['env'])
    network = builder.build_network(config['network'])
    loop = builder.build_strategy(config['strategy'], env, network, args.generation_num, args.process_num,
                                    args.eval_ep_num, args.log, args.save_model_period)
    loop.run()


if __name__ == "__main__":
    main()
