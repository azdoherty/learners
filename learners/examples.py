import gym
import numpy as np
from gym import wrappers
from matplotlib import pyplot as plt
import time
import pandas as pd
import os
import learners

class learnerTest():
    def __init__(self, envType, learnerClass, alpha, gamma, rdr, rdrdecay, verbose=False, filename=None):
        self.envType = envType
        self.env = gym.make(envType)
        self.env.render()
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
        self.learner = learnerClass(self.n_states, self.n_actions, self.alpha, self.gamma)

        # data tracking and statistics
        self.eps_avg_reward = None
        self.eps_explore_probability = None
        self.steps_to_solve = None
        self.q_update = None


    def train(self, episodes, steps):
        """
        :param episodes: number of times to start over
        :param steps: number of steps to take before starting over (early termination if stuck)
        :return:
        """
        self.eps_avg_reward = np.zeros(episodes)
        self.eps_explore_probability = np.zeros(episodes)
        self.steps_to_solve = np.zeros(episodes)
        self.q_update = np.zeros(episodes)

        for i in range(episodes):
            state = self.env.reset()
            oldtable = np.copy(self.learner.qtable)
            for j in range(steps):
                if self.verbose:
                    self.env.render()
                action = self.learner.predict(state)
                newState, reward, done, info = self.env.step(action)
                self.learner.update(state, action, reward, newState, done)
                state = newState
                self.eps_avg_reward[i] += reward
                if done:
                    if self.verbose:
                        print("Episode {} complete after {} steps".format(i, j))
                    break
            self.steps_to_solve[i] = j
            self.eps_explore_probability[i] = self.learner.random_rate
            self.learner.random_rate *= self.rdrdecay
            self.q_update[j] = np.mean(np.sum((oldtable - self.learner.qtable)**2))
        return self.eps_avg_reward

    def plot_rewards(self, window=100, plotFile=None):
        rewardPlot = plt.figure()
        ser = pd.Series(self.eps_avg_reward)
        rolling_avg = ser.rolling(center=False, window=100).mean()
        best_avg = rolling_avg.max()
        #plt.plot(self.eps_avg_reward, label="average reward")
        plt.plot(rolling_avg, label="{} window rolling average".format(window))
        plt.xlabel("episode")
        plt.ylabel("average reward")
        plt.title("{:.2f} Best 100 Episode Average".format(best_avg))
        plt.legend()
        if plotFile:
            plt.savefig(plotFile)
        plt.close()

    def plot_stats(self, plotFile=None):
        step_plot = plt.figure()
        plt.plot(self.steps_to_solve)
        plt.xlabel("Episode")
        plt.ylabel("steps")
        plt.title("Steps To Solve")
        if plotFile:
            plt.savefig(plotFile)
        plt.close()

    def plot_updates(self, plotFile=None):
        step_plot = plt.figure()
        plt.semilogy(self.q_update)
        plt.xlabel("Episode")
        plt.ylabel("Delta Q")
        plt.title("Delta Q per Episode")
        if plotFile:
            plt.savefig(plotFile)
        plt.close()

if __name__ == "__main__":
    if not os.path.exists("output"):
        os.mkdir("output")
    env = "Taxi-v2"
    learner = learners.Qlearner
    episodes = 5000
    steps = 200
    alpha = .5
    gamma = .2
    random_rate = .999
    random_decay_rate = .999
    verbose = True
    example = learnerTest(env, learner, alpha, gamma, random_rate, random_decay_rate, verbose=False)
    example.train(episodes, steps)
    example.plot_rewards(100, os.path.join("output", "rewards.png"))
    example.plot_stats(os.path.join("output", "steps.png"))
    example.plot_updates(os.path.join("output", "q.png"))
    print(example.q_update)



