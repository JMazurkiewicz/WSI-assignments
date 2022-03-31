from dataclasses import dataclass
import random
import numpy as np

@dataclass
class GenerationStats:
    mean: float = 0
    stddev: float = 0
    best: int = 0

class GeneticAlgorithm:
    def __init__(self, fitness_fn, initial_pop, crossover_prob: float, mut_prob: float, iteration_limit: int):
        self.fitness_fn = fitness_fn
        self.population = initial_pop
        self.stats = []
        self.fitnesses = self._calculate_fitnesses_for_current_population()
        self.crossover_prob = crossover_prob
        self.mut_prob = mut_prob
        self._do_evolution(iteration_limit)

    def _calculate_fitnesses_for_current_population(self):
        return [self.fitness_fn(i) for i in self.population]

    def _do_evolution(self, iteration_limit):
        for i in range(iteration_limit):
            print(f'Generation {i + 1}...')
            self._cache_generation_stats()
            self.fitnesses = self._calculate_fitnesses_for_current_population()
            selected = self._do_roulette_selection()
            crossedover = self._do_crossover(selected)
            self.population = self._do_mutations(crossedover)

    def _do_roulette_selection(self):
        fitness_sum = sum(self.fitnesses)
        population_size = len(self.population)
        selection_prob = [self._calc_selection_prob(i, fitness_sum) for i in range(population_size)]
        return random.choices(self.population, weights=selection_prob, k=population_size)

    def _calc_selection_prob(self, individual_index, fitness_sum):
        prob = self.fitnesses[individual_index] / fitness_sum
        return prob if prob > 0 else 0.0

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
        self.stats += [stats]

    def get_final_population(self):
        return self.population

    def dump_final_population(self, file_name='final_population.log'):
        with open(file_name, 'w') as file:
            for i, individual in enumerate(self.population):
                file.write(f'{i + 1:>5}) {individual}\n\n')

    def dump_final_fitnesses(self, file_name='final_fitnesses.log'):
        with open(file_name, 'w') as file:
            for i, fitness in enumerate(self.fitnesses):
                file.write(f'{i + 1:>5}) {fitness}\n')
            file.write(f'mean:   {self.stats[-1].mean}\n')
            file.write(f'stddev: {self.stats[-1].stddev}\n')

    def dump_generation_history(self, file_name='history.log'):
        with open(file_name, 'w') as file:
            for i, stat in enumerate(self.stats):
                file.write(f'{i + 1:>5}) mean={stat.mean}, stddev={stat.stddev:.3f}, best={stat.best}\n')
