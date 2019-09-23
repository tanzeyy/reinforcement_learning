import gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from agents import DDPGAgent
from utils import DDPGReplayBuffer

env_list = [
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

env = gym.make("BipedalWalker-v2")
obs, rew, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0

epochs = 100
steps_per_epoch = 50
max_ep_len = 500
replay_size=int(1e6)
start_steps = 2000
batch_size = 64
tf.set_random_seed(0)

agent = DDPGAgent(env.observation_space, env.action_space)
buffer = DDPGReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], size=replay_size)

rewards = [0]
q_losses = []
pi_losses = []
total_steps = steps_per_epoch * epochs
ep_ret = 0
ep_len = 0
for t in tqdm(range(5000)):
    if t > start_steps:
        act = agent.get_action(obs)
        # print(act)
    else:
        act = env.action_space.sample()

    next_obs, rew, done, _ = env.step(act)
    ep_ret += rew
    ep_len += 1

    done = False if ep_len == max_ep_len else done

    buffer.store(obs, act, rew, next_obs, done)
    obs = next_obs

    if done or (ep_len==max_ep_len):
        rewards.append(ep_ret)
        for _ in range(ep_len):
            batch = buffer.sample_batch(batch_size)
            q_loss, pi_loss = agent.update(batch)
            q_losses.append(q_loss)
            pi_losses.append(pi_loss)
        obs, rew, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0


for i in range(10):
    obs, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    rewards = []
    while not d or ep_len == max_ep_len:
        agent.noise_scale = 0
        act = agent.get_action(obs)
        # print(act)
        obs, r, d, _ = env.step(act)
        ep_len += 1
        ep_ret += r
        rewards.append(r)
        env.render()
    obs, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    print(np.mean(np.array(rewards)))

print(rewards)

np.save("rewards", np.array(rewards))
np.save("q_loss", np.array(q_losses))
np.save("pi_loss", np.array(pi_losses))
