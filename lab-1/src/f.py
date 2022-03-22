# Author: Jakub Mazurkiewicz

import sys
from gradient import GradientDescent
import matplotlib.pyplot as plt
import numpy as np

def f(x: float) -> float:
    return x ** 4

def nabla_f(x: float) -> float:
    return 4 * x ** 3

def plot_f_function_with_points(points, ax):
    x = np.linspace(-15, 15, 10000)
    ax.plot(x, f(x))
    for point in points:
        ax.plot(point, f(point), marker='o', markersize=5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

def plot_f_value_changes(points, ax):
    ax.plot(range(len(points)), [f(x) for x in points])
    ax.set_xlabel('Nr iteracji')
    ax.set_ylabel('Wartości funkcji f')

def make_plots(points):
    fig, axes = plt.subplots(2)
    fig.suptitle('Wyniki eksperymentu')
    plot_f_function_with_points(points, axes[0])
    plot_f_value_changes(points, axes[1])
    plt.show()

def test_gradient(start_point, step_size, max_iteration):
    print(f'Start point -> {start_point}')
    print(f'Step size -> {step_size}')
    print(f'Max iterations -> {max_iteration}')

    grad = GradientDescent(nabla_f, start_point, step_size, max_iteration)
    local_min = grad.get_local_min()
    print(f'Found local minimum -> {local_min:.4f}')
    print(f'Function value in minimum -> {f(local_min):.4f}')
    make_plots(grad.get_points())

def main():
    argv = sys.argv[1:]
    if len(argv) == 3:
        start_point = float(argv[0])
        step_size = float(argv[1])
        max_iteration = int(argv[2])
        test_gradient(start_point, step_size, max_iteration)
    else:
        print(f'Invalid amount of arguments, expected 3 (start point, step size, iteration count) got {len(argv)}.')

if __name__ == '__main__':
    main()
