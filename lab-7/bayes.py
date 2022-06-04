# Author: Jakub Mazurkiewicz
import numpy as np

class NaiveBayes:
    def __init__(self, x_train, y_train):
        sample_count, attrib_count = x_train.shape
        self.classes = np.unique(y_train)
        self.class_count = len(self.classes)

        self.mean = np.zeros((self.class_count, attrib_count))
        self.stdev = np.zeros((self.class_count, attrib_count))
        self.prob = np.zeros([self.class_count])

        for i, c in enumerate(self.classes):
            x_of_c = x_train[y_train == c]
            self.mean[i] = x_of_c.mean(axis=0)
            self.stdev[i] = x_of_c.std(axis=0)
            self.prob[i] = x_of_c.shape[0] / sample_count

    def classify(self, x):
        return [self._classify_one(one) for one in x]

    def _classify_one(self, x):
        search_table = [
            self.prob[i] * np.product(self._pdf_normal_dist(i, x))
            for i in range(self.class_count)
        ]
        predicted = np.argmax(search_table)
        return self.classes[predicted]

    def _pdf_normal_dist(self, class_index, x):
        mean = self.mean[class_index]
        stdev = self.stdev[class_index]
        part1 = 1 / (stdev * np.sqrt(2 * np.pi))
        part2 = np.exp(-0.5 * ((x - mean) / stdev) ** 2)
        return part1 * part2
