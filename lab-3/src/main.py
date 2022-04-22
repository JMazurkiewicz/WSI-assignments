# Author: Jakub Mazurkiewicz
import math
import os
import sys
from time import sleep
from two_player_games.state import State
from two_player_games.player import Player
from random import randint
from two_player_games.connect_four import ConnectFour, ConnectFourMove
from minimax import MinimaxAlgorithm

def heuristic(state: State, player: Player, max_depth: int, depth: int) -> int:
    FACTOR = 1000000
    if state.get_winner() is player:
        return FACTOR * (max_depth - depth)
    else:
        return -FACTOR

def readraw_game_board(game):
    os.system('clear')
    print(game, flush=True)

def announce_winner(game):
    winner = game.get_winner()
    if winner is None:
        print('It\'s a tie!')
    else:
        print(f'Player {winner.char} won!')

def minimax_vs_player(max_depth):
    game = ConnectFour()
    ai_player = game.get_players()[randint(0, 1)]
    algo = MinimaxAlgorithm(game, max_depth, ai_player, heuristic)

    while not game.is_finished():
        readraw_game_board(game)
        if ai_player is game.get_current_player():
            print('Minimax makes a move...')
            game.make_move(algo.get_next_move())
        else:
            while True:
                try:
                    col = int(input('Enter column number 0..6: '))
                    break
                except ValueError:
                    print('Invalid input. Try again...')
            game.make_move(ConnectFourMove(col))

    readraw_game_board(game)
    announce_winner(game)

def minimax_vs_minimax(max_depth1, max_depth2):
    DELAY = 0.8
    game = ConnectFour()

    player1 = game.get_players()[0]
    algo1 = MinimaxAlgorithm(game, max_depth1, player1, heuristic)
    player2 = game.get_players()[1]
    algo2 = MinimaxAlgorithm(game, max_depth2, player2, heuristic)

    while not game.is_finished():
        readraw_game_board(game)
        if player1 is game.get_current_player():
            game.make_move(algo1.get_next_move())
        else:
            game.make_move(algo2.get_next_move())
        sleep(DELAY)
    readraw_game_board(game)
    announce_winner(game)

def minimax_vs_minimax_automatic(max_depth1, max_depth2):
    LOOP_COUNT = 1000
    player1_win_counter = 0
    player2_win_counter = 0
    tie_counter = 0

    for i in range(LOOP_COUNT):
        print(f'Game {i}...')
        game = ConnectFour()
        player1 = game.get_players()[0]
        player2 = game.get_players()[1]
        algo1 = MinimaxAlgorithm(game, max_depth1, player1, heuristic)
        algo2 = MinimaxAlgorithm(game, max_depth2, player2, heuristic)

        while not game.is_finished():
            if game.get_current_player() is player1:
                game.make_move(algo1.get_next_move())
            else:
                game.make_move(algo2.get_next_move())

        if game.get_winner() is player1:
            player1_win_counter += 1
        elif game.get_winner() is player2:
            player2_win_counter += 1
        else:
            tie_counter += 1

    print(f'Player 1 wins: {player1_win_counter} (max depth = {max_depth1})')
    print(f'Player 2 wins: {player2_win_counter} (max depth = {max_depth2})')
    print(f'Ties: {tie_counter}')

def main():
    argv = sys.argv[1:]
    if len(argv) == 3 or (len(argv) == 2 and int(argv[0]) == 0):
        mode = int(argv[0])
        max_depth1 = int(argv[1])
        max_depth2 = int(argv[2]) if len(argv) == 3 else None

        if mode == 0:
            minimax_vs_player(max_depth1)
        elif mode == 1:
            minimax_vs_minimax(max_depth1, max_depth2)
        elif mode == 2:
            minimax_vs_minimax_automatic(max_depth1, max_depth2)
        else:
            print(f'Invalid mode (should be 0, 1 or 2)')
    else:
        print(f'Invalid amount of arguments: expected two or three arguments (mode, depth-limit1 and optional depth-limit2), got {len(argv)}.')

if __name__ == '__main__':
    main()
