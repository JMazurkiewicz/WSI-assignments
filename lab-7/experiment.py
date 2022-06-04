# Author: Jakub Mazurkiewicz
from argparse import ArgumentParser
from random import Random

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

from bayes import NaiveBayes

def get_training_and_validating_sets(data, chunk_size, chunk_to_skip):
    result = []
    skip_pos = chunk_to_skip * chunk_size
    for j in range(0, len(data), chunk_size):
        if skip_pos != j:
            result += data[j:j+chunk_size]
    return np.array(result), np.array(data[skip_pos:skip_pos+chunk_size])

def run_experiment(x, y, chunk_size, chunk_to_skip, test_size):
    x_test = x[:test_size]
    y_test = y[:test_size]
    x_train, x_valid = get_training_and_validating_sets(x[test_size:], chunk_size, chunk_to_skip)
    y_train, y_valid = get_training_and_validating_sets(y[test_size:], chunk_size, chunk_to_skip)
    algo = NaiveBayes(x_train, y_train)
    results = algo.classify(x_valid)

    print(f'Confusion matrix:\n{confusion_matrix(y_valid, results)}')
    print(f'Precision: {list(precision_score(y_valid, results, average=None))}')
    accuracy_valid = accuracy_score(y_valid, results)
    print(f'Accuracy: {100 * accuracy_valid:.2f}%')

    accuracy_test = accuracy_score(y_test, algo.classify(x_test))
    print(f'Accuracy for test set: {100 * accuracy_test:.2f}%')
    return accuracy_valid, accuracy_test

if __name__ == '__main__':
    argparser = ArgumentParser(description='Naive Bayes classifier')
    argparser.add_argument(
        '-s', '--seed', dest='seed', type=int, default=511,
        help=f'Seed for train/test split (default: 511)'
    )
    argparser.add_argument(
        '-k', dest='k', type=int, default=5,
        help=f'Parameter for cross-validation - chunk count (default: 5)'
    )
    argparser.add_argument(
        '-t', '--test-size', dest='test_size', type=int, default=30,
        help=f'Size of testing set (default: 30)'
    )
    args = argparser.parse_args()
    seed = args.seed
    k = args.k
    test_size = args.test_size

    iris = load_iris()
    train_valid_size = len(iris.data) - test_size
    if train_valid_size % k != 0:
        print(f'Please pick `k` such that {train_valid_size} would be divisible '
              f'by it ({train_valid_size} % {k} = {train_valid_size % k})')
    else:
        zipped_iris = list(zip(iris.data, iris.target))
        Random(seed).shuffle(zipped_iris)
        iris.data, iris.target = zip(*zipped_iris)
        chunk_size = train_valid_size // k

        print(f'Training set size:   {(k - 1) * chunk_size}')
        print(f'Validating set size: {chunk_size}')
        print(f'Testing set size :   {test_size}')
        results_valid = []
        results_test = []
        for i in range(k):
            print('{:=^45}'.format(f' k = {i} '))
            accuracy_valid, accuracy_test = run_experiment(iris.data, iris.target, chunk_size, i, test_size)
            results_valid.append(accuracy_valid)
            results_test.append(accuracy_test)
        print('{:=^45}'.format(' SUMMARY '))
        print(f'Average accuracy (validating set): {100 * np.mean(results_valid):.2f}%')
        print(f'Average accuracy (testing set): {100 * np.mean(results_test):.2f}%')
