import hydra
from omegaconf import DictConfig, OmegaConf

from learning_strategies import Gaussian, RandomAction


@hydra.main(config_path="conf", config_name="fake_config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    if cfg.learning_strategy == "gaussian":
        ls = Gaussian(cfg.env, cfg.network, cfg.cpu_num)
    ls.run()


if __name__ == "__main__":
    main()
