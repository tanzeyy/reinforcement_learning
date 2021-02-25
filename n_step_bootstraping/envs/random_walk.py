import gym
import random
from gym import spaces

class RandomWalkEnv(gym.Env):
    '''Simple random walk environment
    A simple random walk env that is a Markov reward process.
    '''
    def __init__(self, num_states=5):
        self.action_space = None
        self.observation_space = spaces.Discrete(num_states)
        self.n = num_states

    def reset(self):
        self.state = self.n // 2
        return self.state

    def step(self, action=None):
        if not action:
            if random.random() > 0.5:
                self.state += 1
            else:
                self.state -= 1
        
        # Check state
        if self.state == -1:
            reward = -1
            done = True
        elif self.state == self.n:
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        return self.state, reward, done, {}


if __name__ == "__main__":
    env = RandomWalkEnv(5)
    obs = env.reset()

    for _ in range(1000):
        next_obs, reward, done, _ = env.step()
        print("{} {} {} {}".format(obs, next_obs, reward, done))
        if done:
            break
        obs = next_obs