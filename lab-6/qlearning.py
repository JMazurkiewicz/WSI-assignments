# Author: Jakub Mazurkiewicz
from dataclasses import dataclass
import random
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from gym import Env

@dataclass
class Hyperparameters:
    learning_rate: float = 0.5
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
    learning_episodes: int
    stats: List[EpisodesStats]

    def __repr__(self):
        return '\n'.join([
            f'After {self.learning_episodes} episodes:',
            f'* Average epochs per episode: {self.avg_epoch_count}',
            f'* Average penalty per episode: {self.avg_penalty}'
        ])

def make_stats_plot(stats: List[EpisodesStats], major_title: str):
    episodes = len(stats)
    fig, axes= plt.subplots(2)
    fig.suptitle(major_title)
    axes[0].scatter(range(1, episodes + 1), [stat.penalty_count for stat in stats], s=3)
    axes[0].set_xlabel('Number of episodes')
    axes[0].set_ylabel('Penalty count')
    axes[0].set_title('Penalties per episode')
    axes[0].grid(True)
    axes[1].scatter(range(1, episodes + 1), [stat.total_reward for stat in stats], s=3)
    axes[1].set_xlabel('Number of episodes')
    axes[1].set_ylabel('Sum of rewards')
    axes[1].set_title('Sum of rewards per episode')
    axes[1].grid(True)

class QLearning:
    def __init__(self, environment: Env, hyperparameters: Hyperparameters):
        self.env = environment
        self.params = hyperparameters
        self.qtable = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.total_learning_episodes = 0

    def learn(self, seed=None) -> Tuple[List[EpisodesStats], List[EvaluationResult]]:
        learning_stats = []
        evaluation_stats = []
        stop_point_for_evaluation = self.params.episodes // 4

        for episode in range(self.params.episodes):
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
            learning_stats += [stats]

            if (episode + 1) % stop_point_for_evaluation == 0:
                evaluation_stats += [self.evaluate()]
            self.total_learning_episodes += 1
        return learning_stats, evaluation_stats

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
            learning_episodes=self.total_learning_episodes,
            stats=stat_list
        )

    def save_qtable(self, filename):
        np.savetxt(filename, self.qtable)
