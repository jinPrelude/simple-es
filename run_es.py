import logging
import random

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from learning_strategies.evolution.loop import ESLoop


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(config_path="conf", config_name="lunar_lander_config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    network = instantiate(cfg.network)
    offspring_strategy = instantiate(cfg.offspring_strategy)
    env = instantiate(cfg.env)
    ls = ESLoop(
        offspring_strategy,
        env,
        network,
        cfg.generation_num,
        cfg.cpu_num,
        cfg.eval_ep_num,
        cfg.log,
    )
    ls.run()


if __name__ == "__main__":
    main()
