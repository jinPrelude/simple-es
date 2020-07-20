from simple_es.es import ES
from simple_es.agent import Agent
import gym
import simple_es
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="./configs/lunarlander.yaml")
def main(cfg: DictConfig = None):
    env = gym.make(cfg.env_name)
    es = ES(
        epoch=cfg.epoch,
        env=env,
        max_episode_step=cfg.max_episode_step,
        seed=cfg.seed,
        model=Agent,
        population_size=cfg.population_size,
        num_process=cfg.num_process,
        target_reward=cfg.target_reward,
        wandb_log=cfg.wandb_log,
    )
    es.run()


if __name__ == "__main__":
    main()
