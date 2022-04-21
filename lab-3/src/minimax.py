# Author: Jakub Mazurkiewicz
from two_player_games.game import Game
from two_player_games.state import State

class MinimaxAlgorithm:
    def __init__(self, game: Game, depth: int):
        self.game = game
        self.depth = depth

    def find_next_move(self, state):
        pass

    def is_terminal(self, state):
        pass
