# Author: Jakub Mazurkiewicz
from dataclasses import dataclass
import random
from time import sleep
from typing import List
import numpy as np
from gym import Env

@dataclass
class Hyperparameters:
    learning_rate: float = 0.1
    discount_factor: float = 0.6
    epsilon: float = 0.1
    episodes: int = 1000

    def __repr__(self):
        return '\n'.join([
            f'Learning rate: {self.learning_rate}',
            f'Discount factor: {self.discount_factor}',
            f'Epsilon: {self.epsilon}',
            f'Episodes: {self.episodes}'
        ])

@dataclass
class LearningStats:
    penalty_count: int = 0
    total_reward: int = 0

@dataclass
class EvaluationStats:
    avg_epoch_count: float = 0
    avg_penalty: float = 0

class QLearning:
    def __init__(self, environment: Env, hyperparameters: Hyperparameters):
        self.env = environment
        self.params = hyperparameters
        self.qtable = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def learn(self, seed=None) -> List[LearningStats]:
        stat_list = []
        for _ in range(self.params.episodes):
            stats = LearningStats()
            seed = 0xABBA if seed is None else random.getrandbits(64)
            state = self.env.reset()
            done = False

            while not done:
                action = self._epsilon_greedy_exploration(state)
                next_state, reward, done, _ = self.env.step(action)
                self.qtable[state, action] = self.qtable[state, action] + self.params.learning_rate * (
                    reward + self.params.discount_factor * np.max(self.qtable[next_state]) - self.qtable[state, action])
                state = next_state
                stats.total_reward += reward
                stats.penalty_count += 1 if reward < 0 else 0
            stat_list += [stats]
        return stat_list

    def _epsilon_greedy_exploration(self, state):
        if np.random.random() < self.params.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.qtable[state])

    def evaluate(self) -> EvaluationStats:
        total_epochs = 0
        total_penalties = 0

        for _ in range(self.params.episodes):
            state = self.env.reset()
            epochs = 0
            penalties = 0
            reward = 0
            done = False
            while not done:
                action = np.argmax(self.qtable[state])
                state, reward, done, _ = self.env.step(action)
                if reward < 0:
                    penalties += 1
                epochs += 1

            total_epochs += epochs
            total_penalties += penalties
        return EvaluationStats(
            avg_epoch_count=(total_epochs / self.params.episodes),
            avg_penalty=(total_penalties / self.params.episodes)
        )

    def play(self, seed=0xDEADBEEF, delay=1.0):
        state = self.env.reset(seed=seed)
        done = False
        while not done:
            self.env.render()
            action = np.argmax(self.qtable[state])
            state, _, done, _ = self.env.step(action)
            sleep(delay)
        self.env.render()

    def load_qtable(self, filename):
        self.qtable = np.loadtxt(filename)

    def save_qtable(self, filename):
        np.savetxt(filename, self.qtable)
