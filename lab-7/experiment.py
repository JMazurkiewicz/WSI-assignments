# Author: Jakub Mazurkiewicz
from argparse import ArgumentParser
from random import Random

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

from bayes import NaiveBayes

def compose_train_set(data, chunk_size, chunk_to_skip):
    result = []
    for j in range(0, len(data), chunk_size):
        if chunk_to_skip != j:
            result += data[j:j+chunk_size]
    return np.array(result)

def run_experiment(x, y, chunk_size, chunk_to_skip, test_size):
    x_test = x[:test_size]
    y_test = y[:test_size]
    x = x[test_size:]
    y = y[test_size:]

    x_train = compose_train_set(x, chunk_size, chunk_to_skip)
    y_train = compose_train_set(y, chunk_size, chunk_to_skip)
    algo = NaiveBayes(x_train, y_train)
    x_valid = x[chunk_to_skip:chunk_to_skip+chunk_size]
    y_valid = y[chunk_to_skip:chunk_to_skip+chunk_size]
    results = algo.classify(x_valid)

    print(f'Confusion matrix:\n{confusion_matrix(y_valid, results)}')
    print(f'Precision: {list(precision_score(y_valid, results, average=None))}')
    accuracy_valid = accuracy_score(y_valid, results)
    print(f'Accuracy: {100 * accuracy_valid:.2f}%')
    accuracy_test = accuracy_score(y_test, algo.classify(x_test))
    print(f'Accuracy of test set: {100 * accuracy_test:.2f}%')
    return accuracy_valid

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

    iris = load_iris()
    iris_size = len(iris.data)
    if iris_size % args.k != 0:
        print(f'Please pick `k` such that {iris_size} would be divisible '
              f'by it ({iris_size} % {args.k} = {iris_size % args.k})')
    else:
        zipped_iris = list(zip(iris.data, iris.target))
        Random(args.seed).shuffle(zipped_iris)
        iris.data, iris.target = zip(*zipped_iris)
        chunk_size = iris_size // args.k

        results = []
        for i in range(args.k):
            print(f'==============> {i}')
            result = run_experiment(iris.data, iris.target, chunk_size, i, args.test_size)
            results.append(result)

        print(f'Average accuracy: {100 * np.mean(results):.2f}%')
