# Author: Jakub Mazurkiewicz
from two_player_games.state import State
from two_player_games.player import Player

def heuristic(state: State, depth: int, player: Player) -> int:
    return 10000 - depth
