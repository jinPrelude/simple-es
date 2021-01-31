import os
from copy import deepcopy

import hydra
import numpy as np
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="bipedalwalker_config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    for _ in range(100):
        env = hydra.utils.instantiate(cfg.env)
        network = hydra.utils.instantiate(cfg.network)

        save_dir = "outputs/2021-01-29/21-07-14/saved_models/ep_150"
        save_dir = to_absolute_path(save_dir)
        model_list = os.listdir(save_dir)
        models = {}
        for k in model_list:
            network.load_state_dict(torch.load(os.path.join(save_dir, k)))
            models[k] = deepcopy(network)
            models[k].eval()
        obs = env.reset()
        hidden_states = {}
        for k, model in models.items():
            hidden_states[k] = model.init_hidden()

        done = False
        episode_reward = 0
        ep_step = 0
        while not done:
            actions = {}
            with torch.no_grad():
                # ray.util.pdb.set_trace()
                for k, model in models.items():
                    s = torch.from_numpy(obs[k]["state"][np.newaxis, ...]).float()
                    actions[k], hidden_states[k] = model(s, hidden_states[k])
            obs, r, done, _ = env.step(actions)
            env.render()
            episode_reward += r
            ep_step += 1
        print("reward: ", episode_reward, "ep_step: ", ep_step)
        env.close()


if __name__ == "__main__":
    main()
