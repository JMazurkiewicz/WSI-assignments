# Assignment 1

## Goal

Goal: implement descent gradient algorithm.

Score: **7/7**

## About

Programs `f.py` and `g.py` find local minimum of $f$ and $g$ functions:

* $f(x) = x^4$
* $g(x) = 1.5 - \exp(-x_1^2 - x_2^2) - 0.5\exp(-(x_1 - 1)^2 - (x_2 + 2)^2)$

## How to run

```bash
python f.py <starting-point> <step-size> <iteration-limit>
python g.py <x> <y> <step-size> <iteration-limit> # starting-point = (x, y)
```

## Notes

File `documentation.pdf` contains documentation in Polish language.
