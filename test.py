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
from moviepy.editor import ImageSequenceClip


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default="conf/ant.yaml")
    parser.add_argument("--ckpt-path", type=str)
    parser.add_argument("--save-gif", action="store_true")
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    env = builder.build_env(config["env"])
    agent_ids = env.get_agent_ids()

    if args.save_gif:
        run_num = args.ckpt_path.split("/")[-3]
        save_dir = f"test_gif/{run_num}/"
        os.makedirs(save_dir)

    network = builder.build_network(config["network"])
    network.load_state_dict(torch.load(args.ckpt_path))
    for i in range(100):
        models = {}
        for agent_id in agent_ids:
            models[agent_id] = deepcopy(network)
            models[agent_id].eval()
            models[agent_id].reset()
        obs = env.reset()

        done = False
        episode_reward = 0
        ep_step = 0
        ep_render_lst = []
        while not done:
            actions = {}
            for k, model in models.items():
                s = obs[k]["state"][np.newaxis, ...]
                actions[k] = model(s)
            obs, r, done, _ = env.step(actions)
            rgb_array = env.render()
            if args.save_gif:
                ep_render_lst.append(rgb_array)
            episode_reward += r
            ep_step += 1
        print("reward: ", episode_reward, "ep_step: ", ep_step)
        if args.save_gif:
            clip = ImageSequenceClip(ep_render_lst, fps=30)
            clip.write_gif(save_dir + f"ep_{i}.gif", fps=30)
        del ep_render_lst


if __name__ == "__main__":
    main()
