# Author: Jakub Mazurkiewicz
import sys
from random import randint
from time import sleep
from two_player_games.game import Game
from two_player_games.connect_four import ConnectFour, ConnectFourMove
from minimax import MinimaxAlgorithm

def redraw_game(game: Game):
    print(f'\x1B[2J{game}', flush=True)

def test_minimax(depth):
    game = ConnectFour()
    algo = MinimaxAlgorithm(game, depth)

    # Draw AI player
    ai_player = game.get_players()[randint(0, 1)]

    while not game.is_finished():
        redraw_game(game)
        if game.get_current_player() is not ai_player:
            #TODO player vs player
            column = int(input('TEMP -- enter column number (first column is \'0\'): '))
            game.make_move(ConnectFourMove(column))

            #print('Minimax makes a move...')
            #sleep(0.5)
        else:
            column = int(input('Enter column number (first column is \'0\'): '))
            game.make_move(ConnectFourMove(column))

    redraw_game(game)

def main():
    argv = sys.argv[1:]
    if len(argv) == 1:
        depth = int(argv[0])
        test_minimax(depth)
    else:
        print(f'Invalid amount of arguments: expected one argument (depth), got {len(argv)}.')

if __name__ == '__main__':
    main()
