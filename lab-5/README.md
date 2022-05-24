# Assignment 5

## Goal

Goal: implement multilayer perceptron and chosen gradient optimization algorithm

Score: **8/8**

## About

Program `main.py` trains neural network with images of hand-written digits from MNIST set.

This project was co-authored with [Michał Brus](https://github.com/brusmichal). Original repository is available on his [GitHub](https://github.com/brusmichal/neural_network).

## How to run

```bash
python main.py
```

## Extra programs

Programs `mnist8.py` and `mnist28.py` print digits from MNIST set to the terminal:

```bash
# Default value of [image-index] is 0.
python mnist8.py [image-index]
python mnist28.py [image-index]
```

Program `mnist28.py` requires some extra files from [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/) website (they must be decompressed):

* [http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz)
* [http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz)
* [http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz)
* [http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz)

## Notes

File `documentation.ipynb` contains documentation in Polish.
