# Author: Jakub Mazurkiewicz
from dataclasses import dataclass
import csv
import sys
from typing import List
import sklearn.model_selection as sms

@dataclass
class Data:
    """Data for ID3 algorithm"""
    klass: str
    values: List[str]

    def __repr__(self) -> str:
        formatted_values = ', '.join([f'{v:>20}' for v in self.values])
        return f'[{self.klass:<3} -> ({formatted_values})]'

class ReadyDataSets:
    def __init__(self, path, klass_index):
        self.klass_index = klass_index
        self.training_set = []
        self.validating_set = []
        self.testing_set = []
        self._split_data(self._read_from_file(path))

    def get_training_set(self) -> List[Data]:
        return self.training_set

    def get_validating_set(self) -> List[Data]:
        return self.validating_set

    def get_testing_set(self) -> List[Data]:
        return self.testing_set

    def _split_data(self, data: List):
        self.training_set, self.testing_set = sms.train_test_split(data)
        self.training_set, self.validating_set = sms.train_test_split(self.training_set)

    def _read_from_file(self, path) -> List[Data]:
        data = []
        with open(path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                d = Data(row[self.klass_index], [])
                del row[self.klass_index]
                d.values = row
                data.append(d)
        return data

    def __repr__(self):
        bar = f'\n{str():-<120}\n'
        info = [
            (self.training_set, 'Training set'),
            (self.validating_set, "Validating set"),
            (self.testing_set, 'Testing set')
        ]
        newline = '\n'
        return bar.join([f'{i[1]} (size = {len(i[0])}):\n{newline.join([str(item) for item in i[0]])}' for i in info])

def main():
    if len(sys.argv) != 3:
        print('Expected arguments: file name, class column')
    else:
        print(ReadyDataSets(sys.argv[1], int(sys.argv[2])))

if __name__ == '__main__':
    main()
