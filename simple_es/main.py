from simple_es.es import ES
from simple_es.agent import Agent
import gym
import simple_es
import hydra
from omegaconf import DictConfig
import torch
import pickle

@hydra.main(config_path="./configs/lunarlander.yaml")
def main(cfg: DictConfig = None):
    print(cfg.pretty())
    env = gym.make(cfg.env_name)
    es = ES(
        epoch=cfg.epoch,
        env=env,
        is_continuous_action=cfg.is_continuous_action,
        max_episode_step=cfg.max_episode_step,
        seed=cfg.seed,
        std_init=cfg.std_init,
        model=Agent,
        population_size=cfg.population_size,
        num_process=cfg.num_process,
        target_reward=cfg.target_reward,
        wandb_log=cfg.wandb_log,
    )
    if cfg.test:
        param = pickle.load(open("/home/jinprelude/Documents/simple-es/outputs/2020-07-21/11-34-46/best_model.pt", "rb"))
        es._test(param, render=True, print_log=True, test_episode_num=100)
    else:
        es.run()


if __name__ == "__main__":
    main()
