# Author: Jakub Mazurkiewicz

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from gradient import GradientDescent

def _exp1(x):
    return np.exp(-x[0] ** 2 - x[1] ** 2)

def _exp2(x):
    return np.exp(-(x[0] - 1) ** 2 - (x[1] + 2) ** 2)

def g(x):
    return 1.5 - _exp1(x) - 0.5 * _exp2(x)

def nabla_g(x):
    exp1 = _exp1(x)
    exp2 = _exp2(x)
    return [2 * x[0] * exp1 + (x[0] - 1) * exp2, 2 * x[1] * exp1 + (x[1] + 2) * exp2]

def plot_g_function_with_points(points, ax):
    x = np.arange(-5, 5, 0.01)
    y = np.arange(-5, 5, 0.01)
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, g([x, y]), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    for point in points[::10]:
        ax.scatter(point[0], point[1], g(point))

def plot_g_value_changes(points, ax):
    ax.plot(range(len(points)), [g(x) for x in points])
    ax.set_xlabel('Nr iteracji')
    ax.set_ylabel('Wartość funkcji g')

def make_plots(points):
    fig = plt.figure()
    fig.suptitle('Wyniki eksperymentu')
    plot_g_function_with_points(points, fig.add_subplot(2, 1, 1, projection='3d'))
    plot_g_value_changes(points, fig.add_subplot(2, 1, 2))
    plt.show()

def test_gradient(start_point, step_size, max_iteration):
    np.set_printoptions(precision=2)
    print(f'Start point -> ({start_point[0]}, {start_point[1]})')
    print(f'Step size -> {step_size}')
    print(f'Max iterations -> {max_iteration}')

    grad = GradientDescent(nabla_g, start_point, step_size, max_iteration)
    local_min = grad.get_local_min()
    print(f'Found local minimum -> {local_min}')
    print(f'Function value in minimum -> {g(local_min):.2f}')
    make_plots(grad.get_points())

def main():
    argv = sys.argv[1:]
    if len(argv) == 4:
        start_point = [float(argv[0]), float(argv[1])]
        step_size = float(argv[2])
        max_iteration = int(argv[3])
        test_gradient(start_point, step_size, max_iteration)
    else:
        print('Invalid amount of arguments, expected 4 (start x point,'
            f' start y, step size, iteration count) got {len(argv)}.')

if __name__ == '__main__':
    main()
