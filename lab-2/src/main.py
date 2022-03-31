# Author: Jakub Mazurkiewicz
import sys
import rocket_flight as rf
from genetic import GeneticAlgorithm
import matplotlib.pyplot as plt

def make_plots():
    pass

def test_evolution(prob_1):
    print(f'Chance of `1`s in genome: {prob_1}')

    population = rf.create_random_rocket_flights(prob_1)
    algo = GeneticAlgorithm(rf.calc_rocket_flight_fitness, population, 0.5, 0.1, 100) # TODO dane z CMD
    # TODO wykresy i pisanie tabelki do pliku (moduł MarkdownTableWritter???)

def main():
    argv = sys.argv[1:]
    if len(argv) != 0: # TODO zamiast 0 ilość oczekiwanych parametrów
        prob_1 = float(argv[0])
        test_evolution(prob_1)
    else:
        # TODO zamiast 0 ilość oczekiwanych parametrów
        print(f'Invalid amount of arguments: expected 0 arguments, got {len(argv)}.')

if __name__ == '__main__':
    main()
