# Author: Jakub Mazurkiewicz
import numpy as np

class NaiveBayes:
    def __init__(self, x_train, y_train):
        sample_count, feature_count = x_train.shape
        self.classes = np.unique(y_train)
        class_count = len(self.classes)

        self.mean = np.zeros((class_count, feature_count))
        self.var = np.zeros((class_count, feature_count))
        self.priors = np.zeros(class_count)

        for i, c in enumerate(self.classes):
            xc = x_train[y_train == c]
            self.mean[i, :] = xc.mean(axis=0)
            self.var[i, :] = xc.var(axis=0)
            self.priors[i] = xc.shape[0] / sample_count

    def classify(self, x):
        return [self.classify_one(one) for one in x]

    def classify_one(self, x):
        posteriors = []
        for i, c in enumerate(self.classes):
            prior = np.log(self.priors[i])
            posterior = np.sum(np.log(self.pdf(i, x)))
            posteriors.append(prior + posterior)
        return self.classes[np.argmax(posteriors)]

    def pdf(self, class_index, x):
        mean = self.mean[class_index]
        var = self.var[class_index]
        return np.exp(-((x - mean) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)
