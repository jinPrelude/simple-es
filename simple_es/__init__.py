from gym.envs.registration import register
from simple_es import *

register(
    id="Rastrigin-v0", entry_point="simple_es.envs.rastrigin:Rastrigin",
)
