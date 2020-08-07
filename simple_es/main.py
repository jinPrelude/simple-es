import pickle

import gym
import hydra
import pybulletgym
import torch
from omegaconf import DictConfig

import simple_es
from simple_es.agent import Agent
from simple_es.strategies.es import ES


@hydra.main(config_path="./conf/config.yaml")
def main(cfg: DictConfig = None):
    print(cfg.pretty())
    env = gym.make(cfg.env_name)
    es = ES(
        env=env,
        num_process=cfg.num_process,
        seed=cfg.seed,
        wandb_log=cfg.wandb_log,
        hyperparams=cfg,
    )
    if cfg.test:
        param = pickle.load(open(cfg.test_model_dir, "rb",))
        es._test(param, render=True, print_log=True, test_episode_num=100)
    else:
        es.run()


if __name__ == "__main__":
    main()
