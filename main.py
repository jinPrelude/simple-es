import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from learning_strategies import Gaussian


@hydra.main(config_path="conf", config_name="fake_config")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    print(OmegaConf.to_yaml(cfg))
    if cfg.learning_strategy == "gaussian":
        ls = Gaussian(cfg.env, cfg.network, cfg.cpu_num)
    ls.run()


if __name__ == "__main__":
    main()
