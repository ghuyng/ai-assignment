<p align="center" style="font-size:xxx-large">
 Cờ Gánh bot
</p>

# Useful exposed functions and constants for algorithms

-   `board_to_string()` and `print_board()`

-   `all_to_be_captured()`

-   `get_all_legal_moves()`

-   `board_after_move_and_capturing()`

-   `get_winner()`

-   `rate_board()`

# Implement & Test

In `CoGanh.py`: Replace `choose_move_alg1`'s body with your algorithm (optionally `choose_move_alg0`, too).

```bash

python -m main

# With profiler to find out the bottle neck
python -m cProfile -s tottime -m main

```

# References

## Monte-Carlo tree search (implemented by Phat)

-   [code](https://web.archive.org/web/20160308053456/http://mcts.ai/code/python.html)
-   [MCSTs explaination](https://www.youtube.com/watch?v=Fbs4lnGLS8M)

## Minimax

## Alpha-Beta pruning

-   [Wikipedia/Alpha–beta_pruning#Pseudocode](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning#Pseudocode)
