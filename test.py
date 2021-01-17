import os

import numpy as np
import torch

from envs import EatApple
from networks import EatAppleModel

save_dir = "outputs/2021-01-18/00-01-59/saved_models/ep_1"
a1_path = save_dir + "/agent1"
a2_path = save_dir + "/agent2"

model1 = EatAppleModel()
model2 = EatAppleModel()
model1.load_state_dict(torch.load(a1_path))
model2.load_state_dict(torch.load(a2_path))
model1.eval()
model2.eval()
for _ in range(100):
    env = EatApple(random_goal=False)

    (n1, n2) = env.reset()
    n1 = torch.from_numpy(n1[np.newaxis, ...]).float()
    n2 = torch.from_numpy(n2[np.newaxis, ...]).float()
    hidden1 = model1.init_hidden()
    hidden2 = model2.init_hidden()
    d = False
    ep_r = 0
    while not d:
        env.render()
        action1, hidden1 = model1(n1, hidden1)
        action2, hidden2 = model2(n2, hidden2)
        action1 = torch.argmax(action1).detach().numpy()
        action2 = torch.argmax(action2).detach().numpy()
        # action1 = int(input())
        (n1, n2), r, d = env.step([action1, action2])
        n1 = torch.from_numpy(n1[np.newaxis, ...]).float()
        n2 = torch.from_numpy(n2[np.newaxis, ...]).float()
        ep_r += r
    print("reward: ", ep_r)
