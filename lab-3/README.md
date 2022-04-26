# Assignment 3

## Goal

Goal: implement minimax algorithm with alpha-beta pruning.

Score: **6/7**

## About

The goal of the project is to implement minimax algorithm that can play ["Connect Four"](https://en.wikipedia.org/wiki/Connect_Four) game.

## How to run

```bash
python main.py <mode> <depth-limit1> <depth-limit2>
```

Available modes:

* 0 - minimax vs player (`depth-limit2` argument should be omitted)
* 1 - minimax vs minimax
* 2 - minimax vs minimax in loop (for statistics)

## Extra scripts

Scripts from `src/two_player_games` directory come from [`two-player-games`](https://github.com/lychanl/two-player-games) repository by [lychanl](https://github.com/lychanl).
