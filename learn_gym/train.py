import gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from agents import VPGAgent, PPOAgent
from utils import Buffer

env_list = [
    "Acrobot-v1",
    "CartPole-v0",
    "Pendulum-v0",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "BipedalWalker-v2",
    "Humanoid-V1",
    "Riverraid-v0",
    "Breakout-v0",
    "Pong-v0",
    "MsPacman-v0",
    "SpaceInvaders-v0",
    "Seaquest-v0",
    "LunarLanderV2",
    "Reacher-v2",
    "FrozenLake-v0"
]

env = gym.make('MountainCar-v0')
obs, rew, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0

epochs = 10
local_steps_per_epoch = 1000
# tf.set_random_seed(22222)

agent = PPOAgent(env.observation_space, env.action_space)
buffer = Buffer(env.observation_space.shape, env.action_space.shape, size=local_steps_per_epoch)

rewards = [0]
for epoch in tqdm(range(epochs)):
    # print("Epoch {} Reward {}".format(epoch, rewards[-1]))
    for t in range(local_steps_per_epoch):
        act, v_t, logp_pi = agent.get_action(obs)

        buffer.store(obs, act, rew, v_t, logp_pi) # Last var is logpi (not used in vpg)

        obs, rew, done, _ = env.step(act[0])
        ep_ret += rew
        ep_len += 1

        if done or (t==local_steps_per_epoch-1):
            # if not done:
            #     print("WARNING: trajectory cut off by epoch at %d steps." % ep_len)

            last_val = rew if done else v_t
            buffer.finish_path(last_val)

            if done:
                rewards.append(ep_ret)
                obs, rew, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0


    agent.update(buffer.get())

for i in range(10):
    obs, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    rewards = []
    while not d or ep_len == 1000:
        act, _, _ = agent.get_action(obs)
        obs, r, d, _ = env.step(act[0])
        ep_len += 1
        ep_ret += r
        rewards.append(r)
        env.render()
    obs, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    print(np.mean(np.array(rewards)))

print(rewards)

np.save("rewards", np.array(rewards))

