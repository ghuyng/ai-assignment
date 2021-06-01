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


NEIGHBORS_OF = [[generate_all_moves((i, j)) for j in range(5)] for i in range(5)]


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
    """Mutate board argument!!!"""
    x_old, y_old, x_new, y_new = source[0], source[1], dest[0], dest[1]
    board[x_old][y_old], board[x_new][y_new] = (
        board[x_new][y_new],
        board[x_old][y_old],
    )
    return board


def gen_opposite_position_pairs(position: Tuple[int, int]):
    """Return a list of pairs of opposing positions around the argument.
    -> List[Tuple[Tuple[int, int], Tuple[int, int]]]"""
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


def infer_move(old_board, new_board):
    """Return the move which has been played, given old_board and new_board.
    Tuple -> [Tuple, Tuple] | None"""
    changed_positions = [
        (i, j) for i in range(5) for j in range(5) if old_board[i][j] != new_board[i][j]
    ]
    if changed_positions == []:
        return None
    else:
        # We have to compare to 0 only since chess pieces that were "ganh" would have changed their color, too
        # src position: became empty on the new board
        src = [pos for pos in changed_positions if new_board[pos[0]][pos[1]] == 0][0]
        # des position: was empty on the old board
        des = [pos for pos in changed_positions if old_board[pos[0]][pos[1]] == 0][0]
        return (src, des)


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


TEST_BOARD_2 = [
    [0, 1, 0, -1, 0],
    [0, -1, 1, 1, 1],
    [1, -1, 1, -1, 1],
    [-1, 0, 0, 1, 1],
    [-1, 0, 0, 0, 1],
]


def check_luat_mo(old_board, new_board, playing_player):
    """If "luật mở" is executing:
    Return the list of chess pieces which can move in the "mở" position and itself.
    Else return [] and None.
    -> tuple[list[tuple], tuple] | ([], None)
    """
    moved = infer_move(old_board, new_board)
    if moved is not None:
        src, des = moved
        da_bi_ganh = [
            (i, j)
            for i in range(5)
            for j in range(5)
            if old_board[i][j] == -new_board[i][j]
        ]
        # Get all of our pieces which can move into the "mở" position
        possible_pieces_to_move = [
            pos
            for pos in NEIGHBORS_OF[src[0]][src[1]]
            if new_board[pos[0]][pos[1]] == playing_player
        ]
        if (
            len(da_bi_ganh) == 0
            and len(possible_pieces_to_move) > 0
            and len(ganh(src, new_board, playing_player)) > 0
        ):
            return (possible_pieces_to_move, src)
    return ([], None)


# Used to store the board's information after "our" previous move, to adhere to an "open move"
previous_board = copy.deepcopy(INITIAL_BOARD)


def move(board, player):
    # (board: List[List[int]], player: int)
    # -> Tuple[Tuple[int, int], Tuple[int, int]] | None
    # TODO: insert something useful here

    possible_pieces_to_move, mo_position = check_luat_mo(
        old_board=previous_board, new_board=board, playing_player=player
    )

    if mo_position is not None and len(possible_pieces_to_move) > 0:
        # TODO: Decide which one to move
        src = possible_pieces_to_move[0]
        des = mo_position
    else:
        # Dummy moves
        src = (0, 0)
        des = (0, 0)

    new_board = board_after_move(src, des, board)

    previous_board = copy.deepcopy(new_board)

    return None
