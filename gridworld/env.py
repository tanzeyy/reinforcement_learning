import numpy as np
import matplotlib.pyplot as plt


class Grid:
    def __init__(self, size: tuple, special_states: dict):
        self._x, self._y = size
        self.special_states = special_states
        self.move = {'north': self.north_action,
                     'south': self.south_action,
                     'east': self.east_action,
                     'west': self.west_action}
        self.valid_states = [(i, j) for i in range(self._x)
                             for j in range(self._y)]

    @staticmethod
    def north_action(state):
        return (state[0]-1, state[1])

    @staticmethod
    def south_action(state):
        return (state[0]+1, state[1])

    @staticmethod
    def east_action(state):
        return (state[0], state[1]+1)

    @staticmethod
    def west_action(state):
        return (state[0], state[1]-1)

    def reward(self, state: tuple, action: str):
        if state in self.special_states:
            reward = self.special_states[state][0]
            new_state = self.special_states[state][1]
        else:
            new_state = self.move[action](state)
            if new_state not in self.valid_states:
                reward = -1
                new_state = state
            else:
                reward = 0
        return new_state, reward


class Grid2:
    def __init__(self, size: tuple, special_states: dict):
        self._x, self._y = size
        self.special_states = special_states
        self.move = {'north': self.north_action,
                     'south': self.south_action,
                     'east': self.east_action,
                     'west': self.west_action}
        self.valid_states = [(i, j) for i in range(self._x)
                             for j in range(self._y)]

    @staticmethod
    def north_action(state):
        return (state[0]-1, state[1])

    @staticmethod
    def south_action(state):
        return (state[0]+1, state[1])

    @staticmethod
    def east_action(state):
        return (state[0], state[1]+1)

    @staticmethod
    def west_action(state):
        return (state[0], state[1]-1)

    def reward(self, state: tuple, action: str):
        if state in self.special_states:
            reward = self.special_states[state][0]
            new_state = self.special_states[state][1]
        else:
            new_state = self.move[action](state)
            if new_state not in self.valid_states:
                reward = -1
                new_state = state
            else:
                reward = -1
        return new_state, reward


class Agent:
    def __init__(self, valid_states: set, init_state: tuple):
        self.action_set = ['north', 'south', 'east', 'west']
        self.state = init_state
        self.state_value = {valid_state: 0 for valid_state in valid_states}

    @staticmethod
    def random_policy(action_set):
        return np.random.choice(action_set)

    def greedy_policy(self, state, transitions):
        values = {}
        for action in self.action_set:
            next_state = transitions[action](state)
            if next_state not in self.state_value.keys():
                next_state = state
            values[action] = self.state_value[next_state]
        m = max(values.values())
        actions = [k for k, v in values.items() if v == m]
        return actions

    def move(self, policy):
        return policy(self.action_set)

    def update_state_values(self, next_state, reward, gamma=0.9):
        self.state_value[self.state] = gamma * \
            self.state_value[next_state] + reward

    def reset_state_values(self):
        for key in self.state_value.keys():
            self.state_value[key] = 0


def show_values(size, values):
    table = np.zeros(size)
    for state, value in values.items():
        table[state] = value
    # print(table)
    return table


def draw_policy(policy, gridsize=(4, 4), special_states=[]):
    size = gridsize + (1, 1)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()

    ax.set_xticks(np.arange(0,gridsize[0]+1,1))
    ax.set_yticks(np.arange(0,gridsize[1]+1,1))
    plt.grid()
    for state, actions in policy.items():
        if state in special_states:
            continue
        for action in actions:
            if action == "north":
                plt.arrow(state[1]+0.5, state[0]+0.5, 0, -0.3, head_width=0.08, fc='k', ec='k')
            elif action == "south":
                plt.arrow(state[1]+0.5, state[0]+0.5, 0, 0.3, head_width=0.08, fc='k', ec='k')
            elif action == "east":
                plt.arrow(state[1]+0.5, state[0]+0.5, 0.3, 0, head_width=0.08, fc='k', ec='k')
            elif action == "west":
                plt.arrow(state[1]+0.5, state[0]+0.5, -0.3, 0, head_width=0.08, fc='k', ec='k')
        
    plt.show()

if __name__ == "__main__":
    grid = Grid2(size=(4, 4), special_states={(0, 0): (0, (0, 0)),
                                         (3, 3): (0, (3, 3))})
    agent = Agent(grid.valid_states, (1, 1))
    agent.reset_state_values()
    delta = 0
    gamma = 0.9
    for i in range(10):
        delta = 0
        new_state_value = agent.state_value.copy()
        for state in grid.valid_states:
            v = agent.state_value[state]
            backups = []
            for action in agent.action_set:
                p_s_r = 1
                next_state, reward = grid.reward(state, action)
                new_value = p_s_r * (reward + gamma * agent.state_value[next_state])
                backups.append(new_value)
            new_state_value[state] = np.round(max(backups), decimals=1)
            delta = max(delta, abs(v - new_value))
            
        agent.state_value = new_state_value
        print(delta)
        if delta < 0.1:
            break
        print("Step {} -----------------------".format(i+1))
        table = np.around(show_values((4, 4), agent.state_value), decimals=1)
        print(table)