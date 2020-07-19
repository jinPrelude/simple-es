from simple_es.cartpole_es import CartPoleES
import gym
import simple_es
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="./configs/rastrigin.yaml")
def main(cfg: DictConfig = None):
    print(cfg.pretty())
    env = gym.make("CartPole-v0")
    agent = CartPoleES(
        epoch=1000,
        env=env,
        mu_init=4,
        population_size=50,
        sigma_init=0.8,
        num_process=2,
    )
    agent.run()


if __name__ == "__main__":
    main()
