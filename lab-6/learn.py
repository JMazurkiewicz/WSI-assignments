# Author: Jakub Mazurkiewicz
from argparse import ArgumentParser
from itertools import chain
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qlearning as ql

def parse_hyperparameters():
    params = ql.Hyperparameters()
    argparser = ArgumentParser(description='Taxi problem and QLearning algorithm')
    argparser.add_argument(
        '-lr', '--learning-rate', dest='learning_rate', type=float, default=params.learning_rate,
        help=f'Learning rate (default: {params.learning_rate})'
    )
    argparser.add_argument(
        '-df', '--discount-factor', dest='discount_factor', type=float, default=params.discount_factor,
        help=f'Discount factor (default: {params.discount_factor})'
    )
    argparser.add_argument(
        '-e', '--episodes', dest='episodes', type=int, default=params.episodes,
        help=f'Number of episodes (default: {params.episodes})'
    )
    argparser.add_argument(
        '-eps', '--epsilon', dest='epsilon', type=float, default=params.epsilon,
        help=f'Epsilon for epsilon-greedy evaluation (default: {params.epsilon})'
    )
    args = argparser.parse_args()
    params.learning_rate = args.learning_rate
    params.discount_factor = args.discount_factor
    params.epsilon = args.epsilon
    params.episodes = args.episodes
    return params

if __name__ == '__main__':
    # Environment:
    # * Blue letter - passenger pick-up location
    # * Purple letter - passenger's destination
    env = gym.make('Taxi-v3')

    hyperparameters = parse_hyperparameters()
    print(hyperparameters)

    algo = ql.QLearning(env, hyperparameters)
    learning_stats, evaluation_stats = algo.learn()
    ql.make_stats_plot(learning_stats, 'Learning process')
    algo.save_qtable('qtable.log')

    for stat in evaluation_stats:
        episode = stat.learning_episodes + 1
        ql.make_stats_plot(stat.stats, f'Evaluation while learning (after {episode} episodes of learning)')

    episodes = [f'After episode {stats.learning_episodes + 1}' for stats in evaluation_stats]
    avg_epoch_count = [stats.avg_epoch_count for stats in evaluation_stats]
    avg_penalty = [stats.avg_penalty for stats in evaluation_stats]
    summary = np.transpose([episodes, avg_epoch_count, avg_penalty])
    print(pd.DataFrame(summary, columns=['Episode', 'Average epochs per episode', 'Average penalty per episode']))
    plt.show()
