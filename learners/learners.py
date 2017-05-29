import numpy as np


class Qlearner():
    def __init__(self, n_states, n_actions, alpha, gamma, random_rate=.1, random_decay=.999):
        self.qtable = np.random.random((n_states, n_actions))
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.random_rate = random_rate
        self.random_decay = random_decay

    def predict(self, state):
        explore = np.ranmdom.random()
        if explore < self.random_rate:
            return np.random.randint((0, self.n_actions))
        else:
            return np.argmax(self.qtable[state, :])

    def update(self, state, action, reward, newstate, done):
        optimal = np.max(self.qtable[newstate, :])
        if not done:
            #self.qtable[state, action] += self.alpha * ( reward + self.gamma * (optimal - self.qtable[state, action]))
            self.qtable[state, action] = (1 - self.alpha) * self.qtable[state, action] + \
                                         self.alpha * ( reward + self.gamma * optimal - self.qtable[state, action])
        else:
            self.random_rate *= self.random_decay
            self.qtable[state, action] += self.alpha * ( reward - self.qtable[state, action])

    def evaluate_update(self, old):
        res = np.sum((self.qtable - old)**2)
        return res