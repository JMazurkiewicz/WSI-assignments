# Author: Jakub Mazurkiewicz
from argparse import ArgumentParser
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split

from bayes import NaiveBayes

def parse_args():
    argparser = ArgumentParser(description='Naive Bayes classifier')
    argparser.add_argument(
        '-s', '--seed', dest='seed', type=int, default=511,
        help=f'Seed for train/test split (default: 511)'
    )
    args = argparser.parse_args()
    return args.seed

if __name__ == '__main__':
    seed = parse_args()
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=seed)
    bayes = NaiveBayes(x_train, y_train)
    results = bayes.classify(x_test)

    total = 0
    good = 0
    for got, expected in zip(results, y_train):
        if got == expected:
            good += 1
        total += 1

    print(f'% => {good / total:.2f}')
    print("Confusion matrix\n", confusion_matrix(y_test, results))
    print("Precision: ", list(precision_score(y_test, results, average=None)))
    print("Accuracy: ", accuracy_score(y_test, results))
    print("Recall: ", list(recall_score(y_test, results, average=None)))
