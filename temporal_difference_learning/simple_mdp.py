import gym
import numpy as np
from gym import spaces


class SimpleMDP(gym.Env):
    """
    Simple MDP with two states: A and B.
    The agent always starts at state A, with left and right action 
    transits to B and terminal state. At state B, there are many 
    possible actions all of which cause immediate termination with 
    a reward drawn from a normal distribution with mean -0.1 and variance 1.0.

    States: 0: A
            1: B
            2: Terminal
    Actions: 0: left
             1: right
    """
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(3)

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action=None):
        done = False
        if self.state == 0:
            reward = 0
            if action == 0:
                self.state = 1
            elif action == 1:
                self.state = 2
                done = True
        elif self.state == 1:
            reward = np.random.normal(loc=-0.1, scale=1.0)
            self.state = 2
            done = True
        return self.state, reward, done, {}
