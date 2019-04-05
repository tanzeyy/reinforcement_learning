import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

# def record(*args, **kwargs):
#     doll = Doll()
#     for key, value in kwargs.items():
#         setattr(doll, key, value)
#     return doll

def cal_opt_prop(data):
    counts = {e: d[:, :, 0] - d[:, :, 1] for e, d in data.items()}
    opt_action_curves = {e: [len(np.where(cnt == 0)[0]) / len(cnt) for cnt in count]
                         for e, count in counts.items()}
    return opt_action_curves

def cal_reward(data):
    reward_curves = {e: np.mean(d, axis=1)[:, 2] for e, d in data.items()}
    return reward_curves

def plot_curve(data, ax, title, x_label, y_label):
    assert(isinstance(data, dict))

    for name, curve in data.items():
        ax.plot(curve, label=name)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def plot_data(data):
    reward_curves = {e: np.mean(d, axis=1)[:, 2] for e, d in data.items()}
    counts = {e: d[:, :, 0] - d[:, :, 1] for e, d in data.items()}
    opt_action_curves = {e: [len(np.where(cnt == 0)[0]) / len(cnt) for cnt in count]
                         for e, count in counts.items()}

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    plot_curve(reward_curves,
               ax=ax[0],
               title='Average Reward',
               x_label='steps',
               y_label='Avg. Reward')
    plot_curve(opt_action_curves,
               ax=ax[1],
               title='Optimal Action Proportion',
               x_label='steps',
               y_label='Opt. Act. Prop.')
    plt.subplots_adjust(hspace=0.3)
    plt.show()


def train(testbed, *args, steps=1000, **kwargs):
    '''
    Store all the experience, simplify data storing using index
    Elements in [tuple_len] are action, optimal action, reward and expected reward
    '''
    tuple_len = 4
    data_bin = np.zeros((steps, len(testbed), tuple_len))
    pbar = tqdm(range(len(testbed)))
    for i in pbar:
        agent, bandit = testbed[i]
        optimal_action = bandit.optimal_action
        for j in range(steps):
            action, expected_reward = agent.take_action(*args, step=j, **kwargs)
            reward = bandit.noisy_reward(action)
            # j is beyond i for the sack of convience
            data_bin[j][i] = np.array([action, optimal_action, reward, expected_reward])
            agent.update_values(action, reward)
    return data_bin

# if __name__ == "__main__":
#     a = record(action=4, fuck=54)
#     print(a)
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots(1, 2)
    