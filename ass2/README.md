<p align="center" style="font-size:xxx-large">
 Cờ Gánh bot
</p>

# Useful exposed functions and constants for algorithms

-   `INITIAL_BOARD`: use deepcopy to replicate it

-   `board_to_string()` and `print_board()`

-   `all_to_be_captured()`

-   `prev_boards[player_id]`: has 2 slots, each for a player, don't modify this,
    implement your own method of saving previous board(s)

-   `get_all_legal_moves()`

-   `board_after_move_and_capturing()`

-   `get_winner()`

# Implement & Test

In `CoGanh.py`: Replace `choose_move_alg1`'s body with your algorithm (optionally `choose_move_alg0`, too).

```bash

python -m main

# With profiler to find out the bottle neck
python -m cProfile -s tottime -m main

```
