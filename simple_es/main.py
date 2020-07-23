import pickle

import gym
import hydra
import torch
from omegaconf import DictConfig

import simple_es
from simple_es.agent import Agent
from simple_es.es import ES


@hydra.main(config_path="./configs/lunarlander.yaml")
def main(cfg: DictConfig = None):
    print(cfg.pretty())
    env = gym.make(cfg.env_name)
    es = ES(
        env=env,
        is_continuous_action=cfg.is_continuous_action,
        model=Agent,
        num_process=cfg.num_process,
        seed=cfg.seed,
        wandb_log=cfg.wandb_log,
        hyperparams=cfg.hyperparams,
    )
    if cfg.test:
        param = pickle.load(
            open(
                "/home/jinprelude/Documents/simple-es/outputs/2020-07-21/11-34-46/best_model.pt",
                "rb",
            )
        )
        es._test(param, render=True, print_log=True, test_episode_num=100)
    else:
        es.run()


if __name__ == "__main__":
    main()
