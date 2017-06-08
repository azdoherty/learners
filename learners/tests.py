import gym
import numpy as np
from gym import wrappers
from matplotlib import pyplot as plt
import time
import pandas as pd
import shutil


class learnerTest():
    def __init__(self, envType, learnerClass, timeout, alpha, gamma, rdr, rdrdecay):
        self.envType = envType
        self.env = gym.make(envType)
        self.alpha = alpha
        self.gamma = gamma
        self.rdr = rdr
        self.rdrdecay = rdrdecay

        self.timeout = timeout
        try:
            self.n_states =  self.env.observation_space.n
            self.n_actions =  self.env.action_space.n
        except:
            self.n_states = np.inf
            self.n_actions = self.env.action_space.n
        self.learner = learnerClass(self.n_states, self.n_actions)

    def train(self, episodes):

        for i in range(episodes):
            pass


