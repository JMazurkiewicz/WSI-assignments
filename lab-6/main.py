# Author: Jakub Mazurkiewicz
from argparse import ArgumentParser
import random
import gym
import matplotlib.pyplot as plt
import qlearning as ql

def parse_args():
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
    argparser.add_argument(
        '-p', '--play', dest='should_play', action='store_true', default=False,
        help='Should program play the game after learning (default: False)'
    )
    args = argparser.parse_args()
    params.learning_rate = args.learning_rate
    params.discount_factor = args.discount_factor
    params.epsilon = args.epsilon
    params.episodes = args.episodes
    return params, args.should_play

if __name__ == '__main__':
    # Environment:
    # * Blue letter - passenger pick-up location
    # * Purple letter - passenger's destination
    env = gym.make("Taxi-v3")

    hyperparameters, should_play = parse_args()
    print(hyperparameters)
    algo = ql.QLearning(env, hyperparameters)
    learning_stats = algo.learn()

    plt.figure()
    plt.scatter(range(1, hyperparameters.episodes+1), [stat.penalty_count for stat in learning_stats], s=3)
    plt.xlabel('Numer epizodu')
    plt.ylabel('Ilość przyznanych kar')
    plt.title('Ilość przyznanych kar na epizod')
    plt.grid(True)

    plt.figure()
    plt.scatter(range(1, hyperparameters.episodes+1), [stat.total_reward for stat in learning_stats], s=3)
    plt.xlabel('Numer epizodu')
    plt.ylabel('Suma nagród')
    plt.title('Suma nagród w zależności od epizodu')
    plt.grid(True)
    plt.show()

    algo.save_qtable('qtable.log')

    evaluation_stats = algo.evaluate()
    print(f'Average epochs per episode: {evaluation_stats.avg_epoch_count}')
    print(f'Average penalty per episode: {evaluation_stats.avg_penalty}')

    if should_play:
        algo.play(seed=random.getrandbits(64))
