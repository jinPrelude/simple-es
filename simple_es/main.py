from simple_es.es import BasicEs
import gym
import simple_es
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="./configs/rastrigin.yaml")
def main(cfg: DictConfig = None):
    #print(cfg.pretty())
    env = gym.make("Rastrigin-v0")
    agent = BasicEs(
        epoch=1000,
        env=env,
        mu_init=4,
        population_size=6000,
        sigma_init=0.9,
        num_process=2,
    )
    agent.run()


if __name__ == "__main__":
    main()
