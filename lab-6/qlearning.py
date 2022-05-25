# Author: Jakub Mazurkiewicz
from dataclasses import dataclass
from time import sleep
import numpy as np
from gym import Env

@dataclass
class Hyperparameters:
    learning_rate: float = 0.1
    discount_factor: float = 0.6
    epsilon: float = 0.1
    episodes: int = 1000

@dataclass

class QLearning:
    def __init__(self, env: Env, hyperparameters: Hyperparameters):
        self.qtable = np.zeros((env.observation_space.n, env.action_space.n))
        self.env = env
        self.params = hyperparameters

    def learn(self, seed=None):
        for _ in range(self.params.episodes):
            seed = 0xABBA if seed is None else np.random.randint()
            state = self.env.reset()
            done = False
            while not done:
                action = self._epsilon_greedy_exploration(state)
                next_state, reward, done, _ = self.env.step(action)
                self.qtable[state, action] = self.qtable[state, action] + self.params.learning_rate * (
                    reward + self.params.discount_factor * np.max(self.qtable[next_state]) - self.qtable[state, action])
                state = next_state

    def _epsilon_greedy_exploration(self, state):
        if np.random.random() < self.params.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.qtable[state])

    def evaluate(self):
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
        print(f"Results after {self.params.episodes} episodes:")
        print(f'Average timesteps per episode: {total_epochs / self.params.episodes}')
        print(f'Average penalties per episode: {total_penalties / self.params.episodes}')


    def play(self, seed=0xDEADBEEF, delay=1.0):
        state = self.env.reset(seed=seed)
        done = False
        while not done:
            self.env.render()
            action = np.argmax(self.qtable[state])
            state, _, done, _ = self.env.step(action)
            sleep(delay)
