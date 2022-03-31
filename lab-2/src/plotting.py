# Author: Jakub Mazurkiewicz
import numpy as np
import matplotlib.pyplot as plt

def plot_best_individuals(ax):
    data = np.loadtxt('best_for_each_generation.log')
    for experiment in data:
        for i, best in enumerate(experiment):
            ax.plot(i, best, marker='o', markersize=5)
    ax.set_xlabel('Numer generacji')
    ax.set_ylabel('Zysk')

def plot_means(ax):
    data = np.loadtxt('avg_for_each_generation.log')
    means = np.mean(data, axis=0)
    for i, mean in enumerate(means):
        ax.plot(i, mean, marker='o', markersize=5)
    ax.set_xlabel('Numer generacji')
    ax.set_ylabel('Średni zysk ze wszystkich uruchomień')

def plot_avg_of_best(ax):
    data = np.loadtxt('best_for_each_generation.log')
    means = np.mean(data, axis=0)
    for i, avg_of_best in enumerate(means):
        ax.plot(i, avg_of_best, marker='o', markersize=5)
    ax.set_xlabel('Numer generacji')
    ax.set_ylabel('Średni zysk z najlepszych osobników\nze wszystkich uruchomień')

def plot_avg_of_worst(ax):
    data = np.loadtxt('worst_for_each_generation.log')
    means = np.mean(data, axis=0)
    for i, avg_of_worst in enumerate(means):
        ax.plot(i, avg_of_worst, marker='o', markersize=5)
    ax.set_xlabel('Numer generacji')
    ax.set_ylabel('Średni zysk z najgorszych osobników\nze wszystkich uruchomień')

def main():
    fig, axes = plt.subplots(3)
    fig.suptitle('Wyniki eksperymentu')
    plot_means(axes[0])
    plot_avg_of_best(axes[1])
    plot_avg_of_worst(axes[2])
    plt.show()

if __name__ == '__main__':
    main()
