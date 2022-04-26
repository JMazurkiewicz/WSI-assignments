# Assignment 1

## Goal

Goal: implement descent gradient algorithm.

Score: **7/7**

## About

Programs `f.py` and `g.py` find local minimum of $f$ and $g$ functions:

* f(x) = x^4
* g(x1, x2) = 1.5 - exp(-x1^2 - x2^2) - 0.5exp(-(x1 - 1)^2 - (x2 + 2)^2)

## How to run

```bash
python f.py <starting-point> <step-size> <iteration-limit>
python g.py <x> <y> <step-size> <iteration-limit> # starting-point = (x, y)
```
