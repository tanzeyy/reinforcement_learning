import unittest

import numpy as np

from env import BlackJackEnv, Card


class TestBlackJackEnv(unittest.TestCase):
    def test_card(self):
        a = Card('1', 'Diamond')
        b = Card('Jack', 'Heart')
        self.assertIsInstance(sum(map(int, [a, b])), int)

    def test_env(self):
        dealer_policy = lambda x: np.random.choice([0, 1])
        env = BlackJackEnv(dealer_policy)
        for _ in range(10):
            obs = env.reset()
            while True:
                act = np.random.choice(env.action_space)
                _, _, done, _ = env.step(act)
                if done:
                    break


if __name__ == "__main__":
    unittest.main()
