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

def make_g_function_with_points_plot(points, place):
    x = np.arange(-5, 5, 0.01)
    y = np.arange(-5, 5, 0.01)
    x, y = np.meshgrid(x, y)
    z = g([x, y])
    place.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    for point in points:
        place.scatter(point[0], point[1], g(point))

def make_iterations_vs_value_plot(points, place):
    place.plot(range(len(points)), [g(x) for x in points])
    place.set_xlabel('Nr iteracji')
    place.set_ylabel('Wartość funkcji g')
    pass

def make_plots(grad):
    fig = plt.figure()
    fig.suptitle('Wyniki eksperymentu')
    place_3d = fig.add_subplot(2, 1, 1, projection='3d')
    make_g_function_with_points_plot(grad.get_points(),place_3d)
    regular_place = fig.add_subplot(2, 1, 2)
    make_iterations_vs_value_plot(grad.get_points(), regular_place)
    plt.show()

def test_gradient(gradient_params):
    np.set_printoptions(precision=2)

    grad = GradientDescent(nabla_g, gradient_params[0:2], *gradient_params[2:])
    print(f'Start point -> ({gradient_params[0]}, {gradient_params[1]})')
    print(f'Step size -> {gradient_params[2]}')
    print(f'Max iterations -> {gradient_params[3]}')
    print(f'Local minimum -> {grad.get_local_min()}')
    make_plots(grad)

def main():
    argv = sys.argv[1:]
    if len(argv) == 4:
        argv = [float(argv[0]), float(argv[1]), float(argv[2]), int(argv[3])]
        test_gradient(argv)
    else:
        print('Invalid amount of arguments, expected 4 - start x point,'
            f' start y, step size, iteration count (got {len(argv)})')

if __name__ == '__main__':
    main()
