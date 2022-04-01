# Author: Jakub Mazurkiewicz
import sys
import time as tm
import rocket_flight as rf
from genetic import GeneticAlgorithm

NUMBER_OF_RUNS = 30

def test_evolution(prob_1, pop_size, crossover_prob, mut_prob, iter_limit):
    print(f'Chance of `1`s in genome: {prob_1}')
    print(f'Population size: {pop_size}')
    print(f'Probability of crossover: {crossover_prob}')
    print(f'Probability of mutation: {mut_prob}')
    print(f'Number of generations: {iter_limit}')

    best_for_each_generation_f = open('best_for_each_generation.log', 'w')
    avg_for_each_generation_f = open('avg_for_each_generation.log', 'w')
    worst_for_each_generation_f = open('worst_for_each_generation.log', 'w')

    start = tm.time()
    for i in range(NUMBER_OF_RUNS):
        population = rf.create_random_rocket_flights(pop_size, prob_1)
        print(f'Run {i + 1}...')
        algo = GeneticAlgorithm(rf.calc_rocket_flight_fitness, population, crossover_prob, mut_prob, iter_limit)
        algo.dump_final_stats()
        algo.dump_best_for_each_generation(best_for_each_generation_f)
        algo.dump_avg_for_each_generation(avg_for_each_generation_f)
        algo.dump_worst_for_each_generation(worst_for_each_generation_f)
    print(f'Elapsed: {tm.time() - start:.2f}s')

def main():
    argv = sys.argv[1:]
    if len(argv) == 5:
        prob_1 = float(argv[0])
        pop_size = int(argv[1])
        crossover_prob = float(argv[2])
        mut_prob = float(argv[3])
        iter_limit = int(argv[4])
        test_evolution(prob_1, pop_size, crossover_prob, mut_prob, iter_limit)
    else:
        print(f'Invalid amount of arguments: expected 5 arguments, got {len(argv)}.')

if __name__ == '__main__':
    main()
