import numpy as np


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


class Agent:
    def __init__(self, valid_states: set, init_state: tuple):
        self.action_set = ['north', 'south', 'east', 'west']
        self.state = init_state
        self.state_value = {valid_state: 0 for valid_state in valid_states}

    @staticmethod
    def random_policy(action_set):
        return np.random.choice(action_set)

    def move(self, policy):
        return policy(self.action_set)

    def update_state_values(self, next_state, reward, gamma=0.9):
        self.state_value[self.state] = gamma * \
            self.state_value[next_state] + reward


def show_values(size, values):
    table = np.zeros(size)
    for state, value in values.items():
        table[state] = value
    # print(table)
    return table


if __name__ == "__main__":
    grid = Grid(size=(5, 5), special_states={(0, 1): (10, (4, 1)),
                                             (0, 3): (5, (2, 3))})
    agent = Agent(grid.valid_states, (3, 4))

    mem = []
    policy = agent.random_policy
    
    for i in range(5000):
        action = agent.move(policy)
        next_state, reward = grid.reward(agent.state, action)
        mem.append([agent.state, action, reward, next_state])
        agent.state = next_state
        print(agent.state, action, reward)

    gamma = 0.99
    def func(x, y): return x + gamma * y

    def cnt(m, a, pos):
        def elems(e, pos): return [e[i] for i in pos]
        return np.sum([1 for e in m if a == elems(e, pos)])

    for state in grid.valid_states:
        for action in agent.action_set:
            s_a_cnt = cnt(mem, [state, action], [0, 1])
            s_cnt = cnt(mem, [state], [0])
            p_a = float(s_a_cnt / s_cnt)
            ts = [[r, n_s] for s, a, r, n_s in mem if s == state and a == action]
            agent.state_value[state] += p_a * np.sum([r + gamma * agent.state_value[n_s] for r, n_s in ts]) / s_a_cnt
            pass

    show_values((5, 5), agent.state_value)
