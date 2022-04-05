# Author: Jakub Mazurkiewicz
from sys import argv
import numpy as np
import matplotlib.pyplot as plt

def plot_means(ax, fname_prefix=''):
    data = np.loadtxt(f'{fname_prefix}avg_for_each_generation.log')
    means = np.mean(data, axis=0)
    ax.scatter(range(len(means)), means)
    ax.plot(range(len(means)), means)
    ax.set_xlabel('Numer generacji')
    ax.set_ylabel('Średni zysk ze wszystkich uruchomień')

def plot_avg_of_best(ax, fname_prefix=''):
    data = np.loadtxt(f'{fname_prefix}best_for_each_generation.log')
    means = np.mean(data, axis=0)
    ax.scatter(range(len(means)), means)
    ax.plot(range(len(means)), means)
    ax.set_xlabel('Numer generacji')
    ax.set_ylabel('Średni zysk z najlepszych osobników\nze wszystkich uruchomień')

def plot_avg_of_worst(ax, fname_prefix=''):
    data = np.loadtxt(f'{fname_prefix}worst_for_each_generation.log')
    means = np.mean(data, axis=0)
    ax.scatter(range(len(means)), means)
    ax.plot(range(len(means)), means)
    ax.set_xlabel('Numer generacji')
    ax.set_ylabel('Średni zysk z najgorszych osobników\nze wszystkich uruchomień')

def main():
    fig, axes = plt.subplots(3)
    fig.suptitle('Wyniki eksperymentu')

    prefix = f'{argv[1]}_' if len(argv) >= 2 else ''
    plot_means(axes[0], prefix)
    plot_avg_of_best(axes[1], prefix)
    plot_avg_of_worst(axes[2], prefix)
    plt.show()

if __name__ == '__main__':
    main()
