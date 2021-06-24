import logging
import random
import yaml
import argparse
import numpy as np
import torch
import builder
import os
from copy import deepcopy
from learning_strategies.evolution.loop import ESLoop


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="conf/lunarlander.yaml")
    parser.add_argument('--ckpt-path', type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    env = builder.build_env(config['env'])
    network = builder.build_network(config['network'])
    
    for _ in range(100):
        model_list = os.listdir(args.ckpt_path)
        models = {}
        for k in model_list:
            key = k.split(".")[0]
            network.load_state_dict(torch.load(os.path.join(args.ckpt_path, k)))
            models[key] = deepcopy(network)
            models[key].eval()
            models[key].reset()
        obs = env.reset()

        done = False
        episode_reward = 0
        ep_step = 0
        while not done:
            actions = {}
            with torch.no_grad():
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
