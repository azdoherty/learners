import numpy as np


class Qlearner():
    def __init__(self, n_states, n_actions, alpha, gamma, random_rate=.1, random_decay=.9):
        """
        Q learner to
        :param n_states: int - number of states in state space
        :param n_actions: int - number of actions available at each state
        :param alpha: float - learning rate
        :param gamma: float - discount factor
        :param random_rate: float - probability of making random action
        :param random_decay: float - amount to decay random rate at end of each episode
        """
        self.qtable = np.random.random((n_states, n_actions))
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.random_rate = random_rate
        self.random_decay = random_decay

    def predict(self, state):
        """
        :param state: int - current state of system
        :return: optimal action with probability 1-random rate
        """
        explore = np.random.random()
        if explore < self.random_rate:
            return np.random.randint(0, self.n_actions)
        else:
            return np.argmax(self.qtable[state, :])

    def update(self, state, action, reward, newstate, done):
        """
        :param state: int - previous state of system
        :param action: int - action taken
        :param reward: float - reward received
        :param newstate: int - new state of system
        :param done: boolean - is learning episode finished
        :return:
        """
        optimal = np.max(self.qtable[newstate, :])
        if not done:
            #self.qtable[state, action] += self.alpha * ( reward + self.gamma * (optimal - self.qtable[state, action]))
            self.qtable[state, action] = (1 - self.alpha) * self.qtable[state, action] + \
                                         self.alpha * (reward + self.gamma * optimal - self.qtable[state, action])
        else:
            self.qtable[state, action] += self.alpha * (reward - self.qtable[state, action])

    def evaluate_update(self, old):
        res = np.sum((self.qtable - old)**2)
        return res