#!/usr/bin/env python3

import numpy as np
from functools import reduce
import operator

import time, copy, random

from typing import List, Tuple

INITIAL_BOARD = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, -1],
    [-1, 0, 0, 0, -1],
    [-1, -1, -1, -1, -1],
]

# Whether the chess piece at [i][j] is able to travel diagonally
DIAGONAL_OR_CROSSES_ONLY = [[(i + j) % 2 == 0 for j in range(5)] for i in range(5)]


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
        if ((i != 0 or j != 0) and (can_go_in_8_directions or i == 0 or j == 0))
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


def gen_opposite_position_pairs(
    position: Tuple[int, int]
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Return a list of pairs of opposing positions around the argument."""
    adjacent_positions = generate_all_moves(position)
    x, y = position

    def opposite_pair(adj_pos):
        opp_of_adj_pos = (2 * x - adj_pos[0], 2 * y - adj_pos[1])
        if not (opp_of_adj_pos in adjacent_positions):
            return None
        else:
            # Avoid repetition of opposing pairs
            is_repeated = adjacent_positions.index(adj_pos) > adjacent_positions.index(
                opp_of_adj_pos
            )
            if is_repeated:
                return None
            else:
                return (adj_pos, opp_of_adj_pos)

    return [
        opposite_pair(pos)
        for pos in adjacent_positions
        if opposite_pair(pos) is not None
    ]


# List of pairs of positions that the chess piece at [i][j] can potentially "ganh", regardless of "colors"
# List[List[ List[Tuple[Tuple,Tuple]] ]]
OPPOSITE_POSITION_PAIRS = [
    [gen_opposite_position_pairs((i, j)) for j in range(5)] for i in range(5)
]


def ganh(new_position, board, player) -> List[Tuple]:
    """Tra ve danh sach cac vi tri quan doi phuong ma neu player di vao new_position thi co the ganh."""
    pairs = OPPOSITE_POSITION_PAIRS[new_position[0]][new_position[1]]
    opponent = -player

    def cap_quan_bi_ganh(pair):
        x0, y0 = pair[0]
        x1, y1 = pair[1]
        if board[x0][y0] == board[x1][y1] == opponent:
            return [pair[0], pair[1]]
        return []

    return reduce(operator.add, [cap_quan_bi_ganh(_) for _ in pairs])


def gen_random_board() -> List[List[int]]:
    test_board_arr = (
        [-1 for _ in range(8)] + [1 for _ in range(8)] + [0 for _ in range(25 - 8 * 2)]
    )
    random.shuffle(test_board_arr)
    return [test_board_arr[5 * i : 5 * i + 5] for i in range(5)]


TEST_BOARD = [
    [0, 1, 0, -1, 0],
    [0, -1, 1, 1, 1],
    [1, -1, -1, -1, 1],
    [-1, 0, 0, 0, 1],
    [-1, 0, 0, 1, -1],
]


# Used to store the board's information after "our" previous move, to adhere to an "open move"
previous_board = None


def move(
    board: List[List[int]], player: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    return None
