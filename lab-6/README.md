# Assignment 6

## Goal

Goal: implement Q-learning algorithm

Score: **7/7**

## About

Program `learn.py` performs training of Q-learning algorithm for `Taxi-v3` problem from `gym` library.

## How to run

### Learning

Program `learn.py` creates Q-table (lookup table for Q-learning algorithm) for `Taxi-v3` problem from `gym` library:

```bash
python learn.py [flags...]
```

Use `-h` flag to get information about all available flags.

### Simulation

Program `play.py` runs simulation of `Taxi-v3` problem:

```bash
python play.py [Q-table-filename]
```

Default value of `Q-table-filename` is `qtable.log`.

## Notes

File `documentation.ipynb` contains documentation in Polish language.
