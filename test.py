import os

import numpy as np
import torch

from envs import EatApple
from networks import EatAppleModel

save_dir = "outputs/2021-01-21/22-41-33/saved_models/ep_6"
model_list = os.listdir(save_dir)
models = {}
for k in model_list:
    model_k = EatAppleModel()
    model_k.load_state_dict(torch.load(os.path.join(save_dir, k)))
    models[k] = model_k
    models[k].eval()
for _ in range(100):
    env = EatApple(random_goal=False)

    states = env.reset()
    hidden_states = {}
    for k, model in models.items():
        hidden_states[k] = model.init_hidden()

    done = False
    episode_reward = 0
    while not done:
        actions = {}
        with torch.no_grad():
            # ray.util.pdb.set_trace()
            for k, model in models.items():
                s = torch.from_numpy(states[k][np.newaxis, ...]).float()
                a, hidden_states[k] = model(s, hidden_states[k])
                actions[k] = torch.argmax(a).detach().numpy()
        states, r, done = env.step(actions)
        env.render()
        # self.env.render()
        episode_reward += r
    print("reward: ", episode_reward)
