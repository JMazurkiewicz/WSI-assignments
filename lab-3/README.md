# Assignment 3

## Goal

Goal: implement minimax algorithm with alpha-beta pruning.

Score: **6/7**

## About

Program `main.py` runs minimax algorithm that plays ["Connect Four"](https://en.wikipedia.org/wiki/Connect_Four) game with player (or with another minimax).

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

## Notes

File `documentation.pdf` contains documentation in Polish language.
