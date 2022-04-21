# Author: Jakub Mazurkiewicz
import sys
from random import randint
from two_player_games.connect_four import ConnectFour, ConnectFourMove
from minimax import MinimaxAlgorithm

def redraw_game(game):
    print(f'\x1B[2J{game}', flush=True)

def announce_winner(winner):
    if winner is None:
        print('It\'s a tie!')
    else:
        print(f'Player {winner.char} won!')

def test_minimax(max_depth):
    game = ConnectFour()

    ai_player = game.get_players()[randint(0, 1)]
    algo = MinimaxAlgorithm(game, max_depth, ai_player)

    while not game.is_finished():
        redraw_game(game)
        if game.get_current_player() is not ai_player:
            print('Minimax makes a move...')
            game.make_move(algo.get_next_move())
        else:
            while True:
                try:
                    col = int(input('Enter column number: '))
                    break
                except ValueError:
                    print('Invalid input. Try again...')
            game.make_move(ConnectFourMove(col))
    redraw_game(game)
    announce_winner(game.get_winner())

def main():
    argv = sys.argv[1:]
    if len(argv) == 1:
        max_depth = int(argv[0])
        test_minimax(max_depth)
    else:
        print(f'Invalid amount of arguments: expected one argument (max depth), got {len(argv)}.')

if __name__ == '__main__':
    main()
