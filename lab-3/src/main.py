# Author: Jakub Mazurkiewicz
import sys
from random import randint
from time import sleep
from two_player_games.connect_four import ConnectFour, ConnectFourMove
from minimax import MinimaxAlgorithm

def redraw_game(game):
    print(f'\x1B[2J{game}', flush=True)

def test_minimax(max_depth):
    game = ConnectFour()

    ai_player = game.get_players()[randint(0, 1)]
    algo = MinimaxAlgorithm(game, max_depth, ai_player)

    while not game.is_finished():
        redraw_game(game)
        if game.get_current_player() is not ai_player:
            print('Minimax makes a move...')
            game.make_move(algo.get_next_move())
            #sleep(0.5)
        else:
            column = input('Enter column number (first column is \'0\'): ')
            game.make_move(ConnectFourMove(int(column)))

    redraw_game(game)

def main():
    argv = sys.argv[1:]
    if len(argv) == 1:
        max_depth = int(argv[0])
        test_minimax(max_depth)
    else:
        print(f'Invalid amount of arguments: expected one argument (max depth), got {len(argv)}.')

if __name__ == '__main__':
    main()
