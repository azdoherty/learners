import gym
import numpy as np
from gym import wrappers
from matplotlib import pyplot as plt
import time
import pandas as pd
import shutil
import learners.learners as learners

class learnerTest():
    def __init__(self, envType, learnerClass, alpha, gamma, rdr, rdrdecay, verbose=False,filename=None):
        self.envType = envType
        self.env = gym.make(envType)
        self.filename = filename
        if filename:
            gym.wrappers.Monitor(self.env, filename)

        self.alpha = alpha
        self.gamma = gamma
        self.rdr = rdr
        self.rdrdecay = rdrdecay
        self.verbose = verbose
        try:
            self.n_states =  self.env.observation_space.n
            self.n_actions =  self.env.action_space.n
        except:
            self.n_states = np.inf
            self.n_actions = self.env.action_space.n
        self.learner = learnerClass(self.n_states, self.n_actions)
        self.eps_avg_reward = None

    def train(self, episodes, steps):
        self.eps_avg_reward = np.zeros((episodes))
        for i in range(episodes):
            state = self.env.reset()
            for j in range(steps):
                action = self.learner.predict(state)
                newState, reward, done, info = self.env.step(action)
                self.learner.update(self, state, action, reward, newState, done)
                state = newState
                self.eps_avg_reward[i] += reward
                if done:
                    self.eps_avg_reward[i] /= j
                    if self.verbose:
                        print("Episode {} complete after {} steps".format(i, j))
                    break
        return self.eps_avg_reward

    def plot_rewards(self, plotFile = None):
        rewardPlot = plt.figure()
        plt.plot(self.eps_avg_reward)
        plt.xlabel("episode")
        plt.ylabel("average reward")
        if plotFile:
            plt.savefig(plotFile)
        plt.close()


if __name__ == "__main__":
    env = "FrozenLake8x8-v0"
    learner = learners.Qlearner
    episodes = 1000
    steps = 200
    lakeExample = learnerTest(env, learner, .2, .8, .2, .999)
    lakeExample.train(episodes.steps)



