# Author: Jakub Mazurkiewicz
from dataclasses import dataclass
import random
from typing import List
import numpy as np
from gym import Env

@dataclass
class Hyperparameters:
    learning_rate: float = 0.1
    discount_factor: float = 0.6
    epsilon: float = 0.1
    episodes: int = 10000

    def __repr__(self):
        return '\n'.join([
            f'Learning rate: {self.learning_rate}',
            f'Discount factor: {self.discount_factor}',
            f'Epsilon: {self.epsilon}',
            f'Episodes: {self.episodes}'
        ])

@dataclass
class EpisodesStats:
    penalty_count: int = 0
    total_reward: int = 0

@dataclass
class EvaluationResult:
    avg_epoch_count: float
    avg_penalty: float
    stats: List[EpisodesStats]

class QLearning:
    def __init__(self, environment: Env, hyperparameters: Hyperparameters):
        self.env = environment
        self.params = hyperparameters
        self.qtable = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def learn(self, seed=None) -> List[EpisodesStats]:
        stat_list = []
        for _ in range(self.params.episodes):
            state = self.env.reset() if seed is None else self.env.reset(seed=random.getrandbits(64))
            stats = EpisodesStats()
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

    def evaluate(self, seed=None) -> EvaluationResult:
        total_epochs = 0
        total_penalties = 0
        stat_list = []

        for _ in range(self.params.episodes):
            state = self.env.reset() if seed is None else self.env.reset(seed=random.getrandbits(64))
            stats = EpisodesStats()
            epochs = 0
            reward = 0
            done = False

            while not done:
                action = np.argmax(self.qtable[state])
                state, reward, done, _ = self.env.step(action)
                epochs += 1
                total_penalties += 1 if reward < 0 else 0
                stats.total_reward += reward
                stats.penalty_count += 1 if reward < 0 else 0

            total_epochs += epochs
            stat_list += [stats]

        return EvaluationResult(
            avg_epoch_count=(total_epochs / self.params.episodes),
            avg_penalty=(total_penalties / self.params.episodes),
            stats=stat_list
        )

    def save_qtable(self, filename):
        np.savetxt(filename, self.qtable)
