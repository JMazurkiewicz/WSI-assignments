#import 'two-player-games.two_player_games'
#tpg = __import__('two-player-games.two_player_games.state')
import sys
sys.path.append('two_player_games')
sys.path.append('two-player-games/two_player_games')

import state
import minimax

def main():
    x = state.State(0, 0)

if __name__ == '__main__':
    main()
