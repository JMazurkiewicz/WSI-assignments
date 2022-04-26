# Author: Jakub Mazurkiewicz
from math import inf
import random
from tkinter import W
from two_player_games.game import Game
from two_player_games.move import Move
from two_player_games.player import Player
from two_player_games.state import State
from typing import Tuple

class MinimaxAlgorithm:
    """Minimax algorithm with alpha-beta pruning"""
    def __init__(self, game: Game, heuristic):
        self.game = game
        self.heuristic = heuristic
        self.depth_limit = 0
        self.player = None

    def get_next_move(self, depth_limit: int, player: Player) -> Move:
        if depth_limit == 0:
            return random.choices(self.game.state.get_moves())[0]
        else:
            self.depth_limit = depth_limit
            self.player = player
            return self._alphabeta(self.game.state, self.depth_limit, -inf, inf, True)[1]

    def _random_move(self):
        return

    def _alphabeta(self, node: State, depth: int, alpha: int, beta: int, max_player: bool) -> Tuple[int, Move]:
        value = 0
        next_move = None
        moves = node.get_moves()

        if depth == 0 or node.is_finished():
            return self.default_heuristic(node, depth), next_move
        elif max_player:
            value = -inf
            next_move = random.choice(moves)
            for move in moves:
                child = node.make_move(move)

                new_value = max(value, self._alphabeta(child, depth - 1, alpha, beta, False)[0])
                if new_value > value:
                    value = new_value
                    next_move = move
                elif new_value == value:
                    next_move = next_move if random.random() < 0.5 else move

                alpha = max(alpha, value)
                if value >= beta:
                    break

            return value, next_move
        else:
            value = inf
            next_move = random.choice(moves)
            for move in moves:
                child = node.make_move(move)

                new_value = min(value, self._alphabeta(child, depth - 1, alpha, beta, True)[0])
                if new_value < value:
                    value = new_value
                    next_move = move
                elif new_value == value:
                    next_move = next_move if random.random() < 0.5 else move

                beta = min(beta, value)
                if value <= alpha:
                    break

            return value, next_move

    def default_heuristic(self, node: State, depth: int):
        winner = node.get_winner()
        if winner is self.player:
            return inf
        elif winner is not None:
            return -inf
        else:
            return self.heuristic(node, depth, self.player)
