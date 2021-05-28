#!/usr/bin/env python3

import numpy as np
from functools import reduce
import operator

from typing import List, Tuple

INITIAL_BOARD = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, -1],
    [-1, 0, 0, 0, -1],
    [-1, -1, -1, -1, -1],
]


def board_to_string(board):
    return reduce(
        operator.add,
        [("\t " if i > 0 else "\n") + str(row[i]) for row in board for i in range(5)],
    )


def generate_all_moves(position):
    can_go_in_8_directions = (position[0] + position[1]) % 2 == 0
    unbounded_moves = [
        (position[0] + i, position[1] + j)
        for i in [-1, 0, 1]
        for j in [-1, 0, 1]
        if ((i != 0 or j != 0) and (can_go_in_8_directions or i * j == 0))
    ]
    return [
        (row, col) for (row, col) in unbounded_moves if (0 <= row < 5 and 0 <= col < 5)
    ]


def generate_legal_moves(position, board) -> List[Tuple]:
    moves = generate_all_moves(position)
    return [(i, j) for (i, j) in moves if board[i][j] == 0]


def get_winner(board):
    pO_on_board = True in [(1 in row) for row in board]
    pX_on_board = True in [(-1 in row) for row in board]
    if pO_on_board and not pX_on_board:
        return 1
    if pX_on_board and not pO_on_board:
        return -1
    return 0


def board_after_move(source, dest, board):
    x_old, y_old, x_new, y_new = source[0], source[1], dest[0], dest[1]
    board[x_old][y_old], board[x_new][y_new] = (
        board[x_new][y_new],
        board[x_old][y_old],
    )
    return board


def move(
    board: List[List[int]], player: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    return ((-1, -1), (-1, -1))
