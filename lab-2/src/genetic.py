# Author: Jakub Mazurkiewicz
from dataclasses import dataclass
import random
import numpy as np

@dataclass
class GenerationStats:
    mean: float = 0
    stddev: float = 0
    best: int = 0
    worst: int = 0

class GeneticAlgorithm:
    def __init__(self, fitness_fn, initial_pop, crossover_prob, mut_prob, iteration_limit):
        self.fitness_fn = fitness_fn
        self.population = initial_pop
        self.stats = []
        self._calculate_fitnesses_for_current_population()
        self.crossover_prob = crossover_prob
        self.mut_prob = mut_prob
        self._do_evolution(iteration_limit)

    def _calculate_fitnesses_for_current_population(self):
        self.fitnesses = [self.fitness_fn(i) for i in self.population]

    def _do_evolution(self, iteration_limit):
        print('Evolution', end='')
        for i in range(iteration_limit):
            print('.', end='', flush=True)
            self._cache_generation_stats()
            self._calculate_fitnesses_for_current_population()
            selected = self._do_roulette_selection()
            crossedover = self._do_crossover(selected)
            self.population = self._do_mutations(crossedover)
        print('')
        self._cache_generation_stats()

    def _do_roulette_selection(self):
        fitness_sum = sum(self.fitnesses)
        population_size = len(self.population)
        selection_prob = [max(0.0, self.fitnesses[i] / fitness_sum) for i in range(population_size)]
        return random.choices(self.population, weights=selection_prob, k=population_size)

    def _do_crossover(self, selected):
        crossedover = []
        for i in range(0, len(selected), 2):
            if random.random() < self.crossover_prob:
                crossedover += self._cross_two(selected[i], selected[i + 1])
            else:
                crossedover += selected[i:i + 2]
        return crossedover

    def _cross_two(self, first, second):
        point = random.randint(0, len(first) - 1)
        crossed1 = np.concatenate((first[:point], second[point:]))
        crossed2 = np.concatenate((second[:point], first[point:]))
        assert len(crossed1) == len(crossed2)
        return [crossed1, crossed2]

    def _do_mutations(self, crossedover):
        for individual in crossedover:
            for bit in individual:
                if random.random() < self.mut_prob:
                    bit = 0 if bit != 0 else 1
        return crossedover

    def _cache_generation_stats(self):
        stats = GenerationStats()
        stats.mean = np.mean(self.fitnesses)
        stats.stddev = np.std(self.fitnesses)
        stats.best = max(self.fitnesses)
        stats.worst = min(self.fitnesses)
        self.stats += [stats]

    def dump_final_stats(self, file_name='final_stats.log'):
        with open(file_name, 'w') as file:
            file.write(f'mean:   {self.stats[-1].mean}\n')
            file.write(f'stddev: {self.stats[-1].stddev}\n')
            file.write(f'best:   {self.stats[-1].best}\n')
            file.write(f'worst:  {self.stats[-1].worst}\n')

    def dump_avg_for_each_generation(self, file):
        for stat in self.stats:
            file.write(f'{stat.mean} ')
        file.write('\n')

    def dump_best_for_each_generation(self, file):
        for stat in self.stats:
            file.write(f'{stat.best} ')
        file.write('\n')

    def dump_worst_for_each_generation(self, file):
        for stat in self.stats:
            file.write(f'{stat.worst} ')
        file.write('\n')
