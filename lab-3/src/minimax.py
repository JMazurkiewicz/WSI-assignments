# Author: Jakub Mazurkiewicz
from math import inf
import random
from two_player_games.game import Game
from two_player_games.player import Player
from two_player_games.state import State
class MinimaxAlgorithm:
    """Minimax algorithm with alpha-beta pruning"""
    def __init__(self, game: Game, max_depth: int, player: Player):
        self.game = game
        self.max_depth = max_depth
        self.player = player
        self.choices = None

    def get_next_move(self):
        self.choices = {}
        self._alphabeta(self.game.state, self.max_depth, -inf, inf)
        found_max = max(self.choices.keys())
        return random.choice(self.choices[found_max])

    def _alphabeta(self, state: State, depth: int, alpha: int, beta: int):
        value = 0
        proposed_move = None

        if depth == 0 or state.is_finished():
            value = self._heuristic(state, depth)
        elif state.get_current_player() is self.player:
            value = -inf
            for move in state.get_moves():
                child = state.make_move(move)
                alphabeta = self._alphabeta(child, depth - 1, alpha, beta)
                value = max(value, alphabeta)
                alpha = max(alpha, value)

                if value >= beta:
                    proposed_move = move
                    break
        else:
            value = inf
            for move in state.get_moves():
                child = state.make_move(move)
                alphabeta = self._alphabeta(child, depth - 1, alpha, beta)
                value = min(value, alphabeta)
                beta = min(beta, value)

                if alpha >= beta:
                    proposed_move = move
                    break

        if proposed_move is not None:
            if value not in self.choices:
                self.choices[value] = [proposed_move]
            else:
                self.choices[value] += [proposed_move]
        return value

    def _heuristic(self, state: State, depth: int) -> int:
        FACTOR = inf
        if state.get_winner() is self.player:
            return FACTOR
        else:
            return -FACTOR
