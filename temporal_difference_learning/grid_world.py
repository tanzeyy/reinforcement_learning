import gym
import numpy as np
from gym import spaces


class WindyGridworld(gym.Env):
    """ Windy Gridworld Environment from Example 6.5

    Actions:
        - 0: up
        - 1: down
        - 2: right
        - 3: left
    """
    def __init__(self,
                 world_size=(7, 10),
                 wind_column=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0]):
        self.world = np.zeros(world_size)
        self.world_size = world_size
        for idx, strength in enumerate(wind_column):
            self.world[:, idx] = strength

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete(world_size)

    def reset(self):
        self.pos = np.array([3, 0])  # self.observation_space.sample()
        self.goal = np.array([3, 7])

        return self.pos

    def step(self, action):
        wind = self.world[tuple(self.pos)]
        # Step position
        if action == 0:
            next_pos = self.pos + np.array([-1 - wind, 0])
        elif action == 1:
            next_pos = self.pos + np.array([1 - wind, 0])
        elif action == 2:
            next_pos = self.pos + np.array([-wind, 1])
        elif action == 3:
            next_pos = self.pos + np.array([-wind, -1])
        next_pos_x = np.clip(next_pos[0], 0, self.world_size[0] - 1)
        next_pos_y = np.clip(next_pos[1], 0, self.world_size[1] - 1)
        self.pos = np.array([next_pos_x, next_pos_y], dtype='int')

        if (self.pos == self.goal).all():
            reward = 0
            done = True
        else:
            reward = -1
            done = False

        return self.pos, reward, done, {}


class WindyGridworld_KingsMove(gym.Env):
    """ Windy Gridworld Environment from Example 6.5

    Actions:
        - 0: up
        - 1: down
        - 2: right
        - 3: left
        - 4: up-left
        - 5: up-right
        - 6: down-left
        - 7: down-right
        - 8: stay
    """
    def __init__(self,
                 world_size=(7, 10),
                 wind_column=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
                 stochastic_wind=False):
        self.world = np.zeros(world_size)
        self.world_size = world_size
        for idx, strength in enumerate(wind_column):
            self.world[:, idx] = strength

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.MultiDiscrete(world_size)
        self.stochastic_wind = stochastic_wind

    def reset(self):
        self.pos = np.array([3, 0])  # self.observation_space.sample()
        self.goal = np.array([3, 7])

        return self.pos

    def _compute_pos(self, action):
        wind = self.world[tuple(self.pos)]
        if wind != 0 and self.stochastic_wind:
            wind += np.random.randint(-1, 2)
        if action == 0:  # up
            next_pos = self.pos + np.array([-1 - wind, 0])
        elif action == 1:  # down
            next_pos = self.pos + np.array([1 - wind, 0])
        elif action == 2:  # right
            next_pos = self.pos + np.array([-wind, 1])
        elif action == 3:  # left
            next_pos = self.pos + np.array([-wind, -1])
        elif action == 4:  # up-left
            next_pos = self.pos + np.array([-1 - wind, -1])
        elif action == 5:  # up-right
            next_pos = self.pos + np.array([-1 - wind, 1])
        elif action == 6:  # down-left
            next_pos = self.pos + np.array([1 - wind, -1])
        elif action == 7:  # down-right
            next_pos = self.pos + np.array([1 - wind, 1])
        elif action == 8:  # stay
            next_pos = self.pos + np.array([-wind, 0])
        return next_pos

    def step(self, action):
        # Step position
        next_pos = self._compute_pos(action)
        next_pos_x = np.clip(next_pos[0], 0, self.world_size[0] - 1)
        next_pos_y = np.clip(next_pos[1], 0, self.world_size[1] - 1)
        self.pos = np.array([next_pos_x, next_pos_y], dtype='int')

        if (self.pos == self.goal).all():
            reward = 0
            done = True
        else:
            reward = -1
            done = False

        return self.pos, reward, done, {}


if __name__ == "__main__":
    env = WindyGridworld()
    print(env.world)
    env.reset()
    print(env.pos)
    for i in range(10):
        cur_pos = env.pos
        action = np.random.randint(0, 4)
        env.step(action)
        next_pos = env.pos
        print("Step {}, Pos: {}, Action: {}, Next Pos: {}".format(
            i, cur_pos, action, next_pos))
