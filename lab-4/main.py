# Author: Jakub Mazurkiewicz

import sys
from id3_data import ReadyDataSets
from id3 import Id3

def run(max_depth: int):
    ds = ReadyDataSets('breast-cancer.data', 9) # 9 - irradiat
    id3 = Id3(max_depth)
    id3.train(ds.get_training_set())
    # TODO: get stats, do validation and tests

def main():
    argv = sys.argv[1:]
    if len(argv) == 1:
        max_depth = int(argv[0])
        run(max_depth)
    else:
        print(f'Invalid amount of arguments: expected one argument (max depth), got {len(argv)}.')

if __name__ == '__main__':
    main()
