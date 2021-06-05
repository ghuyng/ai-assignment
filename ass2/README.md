<p align="center" style="font-size:xxx-large">
 **Cờ Gánh bot**
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

Replace `choose_move_alg1`'s body with your algorithm (or `choose_move_alg0`, too).

It's currently unknown how grading will be carried out so we don't define `main()`.

If you don't feel the need to write another file to test, it's possible to run this from the command line:

```bash
python -c 'from CoGanh import * ; simulate()'
```
