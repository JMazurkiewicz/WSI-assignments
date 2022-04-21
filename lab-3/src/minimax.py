# Author: Jakub Mazurkiewicz
from math import inf
from two_player_games.game import Game
from two_player_games.move import Move
from two_player_games.player import Player
from two_player_games.state import State
from typing import Tuple
class MinimaxAlgorithm:
    """Minimax algorithm with alpha-beta pruning"""
    def __init__(self, game: Game, max_depth: int, player: Player):
        self.game = game
        self.max_depth = max_depth
        self.player = player

    def get_next_move(self):
        self._alphabeta(self.game.state, self.max_depth, -inf, inf)

    def _alphabeta(self, state: State, depth: int, alpha: int, beta: int) -> Tuple[int, Move]:
        if depth == 0 or self._is_terminal(state):
            return (self._heuristic(state, depth), None)
        elif state.get_current_player() is self.player:
            for move in state.get_moves():
                child = state.make_move(move)
                alpha = max(alpha, self._alphabeta(child, depth - 1, alpha, beta)[0])
                if alpha >= beta:
                    break
            return alpha
        else:
            for move in state.get_moves():
                child = state.make_move(move)
                beta = min(beta, self._alphabeta(child, depth - 1, alpha, beta))
                if alpha >= beta:
                    break
            return beta

    def _is_terminal(self, state: State):
        return state.is_finished()

    def _heuristic(self, state: State, depth: int) -> int:
        if state.get_winner() is not self.player:
            return -10000 # We've lost!
        elif state.get_winner() is self.player:
            return 1000 * (self.max_depth - depth) # We won (the less moves the better)
        else:
            return (self.max_depth - depth)
