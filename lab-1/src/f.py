# Author: Jakub Mazurkiewicz
import sys
from gradient import GradientDescent
import matplotlib.pyplot as plt
import numpy as np

def f(x: float) -> float:
    return x ** 4

def nabla_f(x: float) -> float:
    return 4 * x ** 3

def make_f_function_with_points_plot(points, place):
    x = np.linspace(-15, 15, 10000)
    y = f(x)
    place.plot(x, y)
    for point in points:
        place.plot(point, f(point), marker='o', markersize=5)
    place.set_xlabel('x')
    place.set_ylabel('y')

def make_iterations_vs_value_plot(points, place):
    place.plot(range(len(points)), [f(x) for x in points])
    place.set_xlabel('Nr iteracji')
    place.set_ylabel('Wartości funkcji f')

def make_plots(grad):
    fig, places = plt.subplots(2)
    fig.suptitle('Wyniki eksperymentu')
    make_f_function_with_points_plot(grad.get_points(), places[0])
    make_iterations_vs_value_plot(grad.get_points(), places[1])
    plt.show()

def test_gradient(gradient_params):
    grad = GradientDescent(nabla_f, *gradient_params)
    print(f'Start point -> {gradient_params[0]}')
    print(f'Step size -> {gradient_params[1]}')
    print(f'Max iterations -> {gradient_params[2]}')

    local_min = grad.get_local_min()
    print(f'Found local minimum -> {local_min:.2f}')
    print(f'Function value in minimum -> {f(local_min):.2f}')

    make_plots(grad)

def main():
    argv = sys.argv[1:]
    if len(argv) == 3:
        argv = [float(argv[0]), float(argv[1]), int(argv[2])]
        test_gradient(argv)
    else:
        print(f'Invalid amount of arguments, expected 3 - start point, step size, iteration count (got {len(argv)})')

if __name__ == '__main__':
    main()
