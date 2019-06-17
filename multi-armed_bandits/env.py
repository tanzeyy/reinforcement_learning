import warnings
from functools import wraps

import numpy as np

import utils


class Env(object):
    def __init__(self, k, mean=0, var=1):
        self._k = k
        self._rewards = self._set_rewards(mean, var)
        self.optimal_action = np.argmax(self._rewards)

    def _set_rewards(self, mean, var):
        return np.random.normal(mean, var, size=(self._k))

    def reward(self, action):
        return self._rewards[action]

    def noisy_reward(self, action):
        return np.random.normal(self._rewards[action], 1)


class NonstationaryEnv(Env):
    def __init__(self, k, mean=0, var=1):
        super().__init__(k, mean, var)

    def _update_rewards(self, mean=0, var=0.01):
        self._rewards += np.random.normal(mean, var, size=(self._k))

    def reward(self, action):
        self._update_rewards()
        return super().reward(action)

    def noisy_reward(self, action):
        self._update_rewards()
        return super().noisy_reward(action)


class Agent(object):
    def __init__(self, k, init_q_value=0):
        self._actions = [i for i in range(k)]
        self._q_values = [init_q_value for _ in range(k)]
        self._occur_times = [0 for _ in range(k)]

    def update_values(self, action, reward):
        '''
        Need to be implemented in subclass.
        '''
        pass

    def take_action(self, *args, **kwargs):
        """
        kwargs:
            c (float): hyper parameter for upper-confidence-bound action selection, 
                        c controls the degree of exploration.
            eps (float): hyper parameter for epsilon-greedy action selection, 
                        if eps is None, enables gradient-ascent-based algo,
                        action-selection obeys to a softmax on q-values of each action.
        Returns:
            action: selected action.
            expected_reward: expected reward from the agent.
        """
        try:
            assert('c' in kwargs)
            assert('eps' in kwargs)
            eps = kwargs.get('eps')
            c = kwargs.get('c')
            step = kwargs.get('step') + 1
            values = self._q_values + c * np.sqrt(
                np.divide(np.log(step), np.array(self._occur_times)))
            if eps != 0:
                warnings.warn(
                    "Value of epsilon is not zero, setting to zero. ")
                eps = 0
        except:
            values = self._q_values

        try:
            assert('eps' in kwargs)  # Epsilon-greedy
            eps = kwargs.get('eps')
            if np.random.uniform(0, 1) > eps:
                action = np.argmax(values)
            else:
                action = np.random.choice(self._actions)
        except:
            # Probability
            action = np.random.choice(self._actions,
                                      p=utils.softmax(self._q_values))
        expected_reward = self._q_values[action]

        self._occur_times[action] += 1

        return action, expected_reward


class SampleAverage(Agent):
    def __init__(self, k, init_q_value=0):
        super().__init__(k, init_q_value)

    def update_values(self, action, reward):
        old_value = self._q_values[action]
        self._q_values[action] = old_value + \
            (reward - old_value) / self._occur_times[action]


class ConstantStepSize(Agent):
    def __init__(self, k, init_q_value=0, alpha=0.1):
        super().__init__(k, init_q_value)
        self._alpha = alpha

    def update_values(self, action, reward):
        old_value = self._q_values[action]
        self._q_values[action] = old_value + self._alpha * (reward - old_value)


class UnbiasedConstantStepSize(Agent):
    def __init__(self, k, init_q_value=0, alpha=0.1):
        super().__init__(k, init_q_value)
        self._alpha = alpha
        # Maintain o-values for each action to calculate step size.
        self._o_values = {i: 0 for i in self._actions}

    def update_values(self, action, reward):
        old_value = self._q_values[action]
        old_o = self._o_values[action]
        self._o_values[action] = old_o + self._alpha * (1 - old_o)
        beta = self._alpha / self._o_values[action]
        self._q_values[action] = old_value + beta * (reward - old_value)


class GradientAscent(Agent):
    def __init__(self, k, init_q_value=0, alpha=0.1):
        super().__init__(k, init_q_value)
        self._alpha = alpha
        self._mean_reward = 0

    def update_values(self, action, reward):
        t = np.sum(self._occur_times)
        self._mean_reward = ((t - 1) * self._mean_reward + reward) / t
        self._q_values -= self._alpha * \
            (reward - self._mean_reward) * utils.softmax(self._q_values)
        self._q_values[action] += self._alpha * (reward - self._mean_reward)


class GradientAscentNoBaseline(Agent):
    def __init__(self, k, init_q_value=0, alpha=0.1):
        super().__init__(k, init_q_value)
        self._alpha = alpha

    def update_values(self, action, reward):
        self._q_values -= self._alpha * reward * utils.softmax(self._q_values)
        self._q_values[action] += self._alpha * reward


class Doll():
    pass


if __name__ == "__main__":
    # from utils import *
    k = 10
    eps = [0, 0.01, 0.1]
    a = Agent(k)
    e = Env(k)
    actions = []
    for i in range(10):
        action, reward = a.take_action(c=2)
        actions.append(action)
        # a.update_values_incrementally
    print(actions)
