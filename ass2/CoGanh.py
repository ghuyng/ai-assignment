#!/usr/bin/env python3

import numpy as np


INITIAL_BOARD = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, -1],
    [-1, 0, 0, 0, -1],
    [-1, -1, -1, -1, -1],
]


def generate_all_moves(position):
    goes_in_eight_directions = (position[0] + position[1]) % 2 == 0
    if goes_in_eight_directions:
        unbounded_moves = [
            (position[0] + i, position[1] + j)
            for i in [-1, 0, 1]
            for j in [-1, 0, 1]
            if (i != 0 or j != 0)
        ]
    else:
        unbounded_moves = [
            (position[0] - 1, position[1]),
            (position[0] + 1, position[1]),
            (position[0], position[1] - 1),
            (position[0], position[1] + 1),
        ]
    return [(i, j) for (i, j) in unbounded_moves if (i >= 0 and j >= 0)]
    # return [
    #     (position[0] + i, position[1] + j)
    #     for i in [-1, 0, 1]
    #     for j in [-1, 0, 1]
    #     if (
    #         (i != 0 or j != 0)
    #         and (position[0] + i >= 0 and position[1] + j >= 0)
    #         and (goes_in_eight_directions or (abs(i) + abs(j) < 2))
    #     )
    # ]


def generate_legal_moves(position, board):
    moves = generate_all_moves(position)
    return [(i, j) for (i, j) in moves if board[i][j] == 0]


def winner(board):
    flatten_board = np.array(board).flatten()
    pO_on_board = 1 in flatten_board
    pX_on_board = -1 in flatten_board
    if pO_on_board and not pX_on_board:
        return 1
    if pX_on_board and not pO_on_board:
        return -1
    return 0


def move(board, player):
    return ((-1, -1), (-1, -1))
