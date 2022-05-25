# Author: Jakub Mazurkiewicz
from argparse import ArgumentParser
import gym
import qlearning as ql

def parse_hyperparameters():
    params = ql.Hyperparameters()
    argparser = ArgumentParser(description='Taxi problem and QLearning algorithm')
    argparser.add_argument('-lr', '--learning-rate',
                           dest='learning_rate', type=float, default=params.learning_rate,
                           help=f'Learning rate (default: {params.learning_rate})')
    argparser.add_argument('-df', '--discount-factor',
                           dest='discount_factor', type=float, default=params.discount_factor,
                           help=f'Discount factor (default: {params.discount_factor})')
    argparser.add_argument('-e', '--episodes',
                           dest='episodes', type=int, default=params.episodes,
                           help=f'Number of episodes (default: {params.episodes})')
    argparser.add_argument('-eps', '--epsilon',
                           dest='epsilon', type=float, default=params.epsilon,
                           help=f'Epsilon for epsilon-greedy evaluation (default: {params.epsilon})')
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
    env = gym.make("Taxi-v3")
    algo = ql.QLearning(env, parse_hyperparameters())
    algo.learn()
    algo.evaluate()
    algo.play()
