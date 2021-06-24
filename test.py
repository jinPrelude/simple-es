import os
from copy import deepcopy

import hydra
import numpy as np
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="simple_spread_config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    env = hydra.utils.instantiate(cfg.env)
    network = hydra.utils.instantiate(cfg.network)
    for _ in range(100):

        ckpt_dir = "outputs/2021-02-02/23-32-46/saved_models/ep_2739"
        ckpt_dir = to_absolute_path(ckpt_dir)
        model_list = os.listdir(ckpt_dir)
        models = {}
        for k in model_list:
            network.load_state_dict(torch.load(os.path.join(ckpt_dir, k)))
            models[k] = deepcopy(network)
            models[k].eval()
            models[k].reset()
        obs = env.reset()

        done = False
        episode_reward = 0
        ep_step = 0
        while not done:
            actions = {}
            with torch.no_grad():
                # ray.util.pdb.set_trace()
                for k, model in models.items():
                    s = torch.from_numpy(obs[k]["state"][np.newaxis, ...]).float()
                    actions[k] = model(s)
            obs, r, done, _ = env.step(actions)
            env.render()
            episode_reward += r
            ep_step += 1
        print("reward: ", episode_reward, "ep_step: ", ep_step)


if __name__ == "__main__":
    main()
