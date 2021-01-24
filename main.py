import logging
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from learning_strategies import Gaussian


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(config_path="conf", config_name="eat_apple_config")
def main(cfg: DictConfig):
    logger = logging.getLogger("logger")
    print(OmegaConf.to_yaml(cfg))

    ls = hydra.utils.instantiate(cfg.learning_strategy, logger=logger)
    ls.run()


if __name__ == "__main__":
    main()
