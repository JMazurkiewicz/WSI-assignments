# Author: Jakub Mazurkiewicz
from dataclasses import dataclass
from id3_data import Data
from math import log
from typing import List

@dataclass
class _Node:
    attrib_index: int
    children: List

    def traverse(self, values: List[str]) -> str:
        if values[self.attrib_index]
        return

@dataclass
class _Leaf():
    klass: str

    def traverse(self, _: List[str]) -> str:
        return self.klass

class _Trainer:
    @staticmethod
    def get_entropy(training_set: List[Data]) -> float:
        all_klasses = [info.klass for info in training_set]
        freq = [all_klasses.count(klass) / len(all_klasses) for klass in set(all_klasses)]
        return -sum(f * log(f) for f in freq)

    @staticmethod
    def get_information_gain(__TTT: int, training_set: List[Data]) -> float:
        return _Trainer.get_entropy(training_set) -

class Id3:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def train(self, training_set: List[Data]):
        all_klasses = [info.klass for info in training_set]

        print(f'Entropy of training set: {_Trainer.get_entropy(training_set)}')

