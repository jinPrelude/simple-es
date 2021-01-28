import json

import gym
import zmq


class UnityEatApple:
    def __init__(self, address, max_step=None):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(address)
        self.max_step = max_step
        self.curr_step = 0
        self.name = "UnityEatApple"

    def reset(self):
        self.curr_step = 0
        return_list = {}
        transition = {}
        # send reset message and get states
        self.socket.send_string("reset")
        rep = self.socket.recv()
        test = {0: 1}
        test = str(test)
        self.socket.send_string(test)
        rep2 = self.socket.recv()
        print(rep2)
        rep2 = json.loads(rep2)
        # "{0:2,1:4}"
        states = json.loads(rep)
        s = self.env.reset()
        # make decoded message to fit our multienv format

        return s

    def step(self, action):
        self.curr_step += 1
        return_list = {}
        transition = {}

        action_json = json.dumps(action)
        rep = self.socket.send_json(action_json)
        rep = self.socket.recv_json()
        obs = json.loads(rep)
        if self.max_step:
            if self.curr_step >= self.max_step or d:
                d = True
        return obs

    def get_agent_ids(self):
        return ["0"]

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


if __name__ == "__main__":
    env = UnityEatApple(address="tcp://localhost:5558", max_step=500)
    for _ in range(50):
        s = env.reset()
        d = False
        ep_r = 0
        while not d:
            n_s, r, d, info = env.step()
            ep_r += r
        print("episode reward: ", ep_r)
