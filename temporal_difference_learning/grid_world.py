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
        # Step position
        if action == 0:
            next_pos = self.pos + np.array(
                [-1 - self.world[tuple(self.pos)], 0])
        elif action == 1:
            next_pos = self.pos + np.array(
                [1 - self.world[tuple(self.pos)], 0])
        elif action == 2:
            next_pos = self.pos + np.array([-self.world[tuple(self.pos)], 1])
        elif action == 3:
            next_pos = self.pos + np.array([-self.world[tuple(self.pos)], -1])
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
