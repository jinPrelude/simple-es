from networks.neural_network import GymEnvModel
from copy import deepcopy
import torch

if __name__ == "__main__":
    a = GymEnvModel()
    a.zero_init()
    a_param = a.get_param_list()
    print(a_param[1])
    b_param = deepcopy(a_param)
    for b in b_param:
        b *= 0
    a.apply_param(b_param)
    chaged_a_param = a.get_param_list()
    print(chaged_a_param[1])
