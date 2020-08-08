import random

import gym
import numpy as np
from gym import spaces


class RacetrackEnv(gym.Env):
    """ Racetrack environment with OpenAI gym interface
    Implementation of racetrack environment, from Exercise 5.12 
    in Sutton and Barto's Reinformcent Learning: An Introduction.
    """
    def __init__(self, track):
        """
        Args:
            track: array representation of the track. Each elements 
                in the array represents:
                - 0: track
                - 1: unreachable area
                - 2: start line
                - 3: finish line
        """
        self.observation_space = spaces.MultiDiscrete(track.shape + (5, 5))
        self.action_space = spaces.MultiDiscrete((3, 3))

        self._start_line = np.array(list(zip(*np.where(track == 2))))
        self._track = track

        self.reset()

    def step(self, action):
        action = np.array(action)
        assert self.action_space.contains(action), "action not valid"
        self._car_vel = self._car_vel + (action - 1)
        self._car_vel = np.clip(self._car_vel, 0, 4)
        self._car_pos = self._car_pos + self._car_vel
        self._car_pos[0] = np.clip(self._car_pos[0], 0,
                                   self._track.shape[0] - 1)
        self._car_pos[1] = np.clip(self._car_pos[1], 0,
                                   self._track.shape[1] - 1)
        x, y = self._car_pos
        next_obs = self._cur_obs
        reward = -1
        done = False
        info = {}
        if self._track[x, y] == 1:
            next_obs = self.reset()
            info['info'] = 'out of boundary'
        elif self._track[x, y] == 3:
            reward = 0
            done = True
            info['info'] = 'reach finish line'
        return next_obs, reward, done, info

    def reset(self):
        self._car_pos = random.choice(self._start_line)
        self._car_vel = np.array([0, 0])

        return self._cur_obs

    @property
    def _cur_obs(self):
        return np.hstack([self._car_pos, self._car_vel])


if __name__ == "__main__":
    from utils import track1
    env = RacetrackEnv(track)
    obs = env.reset()
    env.step([0, 2])
