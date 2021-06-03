#!/usr/bin/env python3

import numpy as np
from functools import reduce
import operator

import time, copy, random

from typing import List, Tuple

from queue import Queue

from copy import deepcopy

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


# Pre-compute for faster access
NEIGHBORS_POSITIONS = [[generate_all_moves((i, j)) for j in range(5)] for i in range(5)]


def generate_legal_moves(position, board) -> List[Tuple]:
    moves = NEIGHBORS_POSITIONS[position[0]][position[1]]
    return [(i, j) for (i, j) in moves if board[i][j] == 0]


def get_winner(board):
    pO_on_board = True in [(1 in row) for row in board]
    pX_on_board = True in [(-1 in row) for row in board]
    if pO_on_board and not pX_on_board:
        return 1
    if pX_on_board and not pO_on_board:
        return -1
    return 0


def board_after_move(source, dest, board, pure=False):
    """If pure is falsy, may mutate board, else return a new one without mutating."""
    if pure:
        new_board = deepcopy(board)
    else:
        new_board = board
    x_old, y_old, x_new, y_new = source[0], source[1], dest[0], dest[1]
    new_board[x_old][y_old], new_board[x_new][y_new] = (
        new_board[x_new][y_new],
        new_board[x_old][y_old],
    )
    return new_board


def board_after_move_and_rules_application(src, des, board, pure=False):
    """If pure is falsy, may mutate board, else return a new one without mutating."""
    player = board[src[0]][src[1]]
    converting = all_to_be_converted(src, des, board, player)
    new_board = board_after_move(src, des, board, pure)
    for (r, c) in converting:
        new_board[r][c] = player
    return new_board


def gen_opposite_position_pairs(position: Tuple[int, int]):
    """Return a list of pairs of opposing positions around the argument.
    -> List[Tuple[Tuple[int, int], Tuple[int, int]]]"""
    adjacent_positions = NEIGHBORS_POSITIONS[position[0]][position[1]]
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


def try_ganh(new_position, board, player) -> List[Tuple]:
    """Tra ve danh sach cac vi tri quan doi phuong ma neu player di vao new_position thi co the ganh.
    board: before executing the move"""
    # TODO: recursive, for example:
    # Player  1  move from  (2, 2)  to  (1, 1) , change the board to:
    # -1       1       -1      0       0
    # 0        1       -1      -1      1
    # 0        1       0       1       1
    # 0        -1      0       1       1
    # 0        -1      -1      0       1
    # Player  -1  move from  (3, 1)  to  (2, 2) , change the board to:
    # -1       1       -1      0       0
    # 0        -1      -1      -1      1
    # 0        -1      -1      -1      1
    # 0        0       0       -1      1
    # 0        -1      -1      0       1
    # 1 at (0,1) should be converted

    pairs = OPPOSITE_POSITION_PAIRS[new_position[0]][new_position[1]]
    opponent = -player

    def cap_quan_bi_ganh(pair):
        x0, y0 = pair[0]
        x1, y1 = pair[1]
        if board[x0][y0] == board[x1][y1] == opponent:
            return [pair[0], pair[1]]
        return []

    return [pos for pair in pairs for pos in cap_quan_bi_ganh(pair)]


def contiguously_surrounded_pieces(initial_pos: Tuple, board) -> List[Tuple]:
    return_value = []
    player = board[initial_pos[0]][initial_pos[1]]
    # Save the traveled positions to avoid repeated traversal
    marked = [[0 for _ in range(5)] for _ in range(5)]
    # Queue to save incoming positions
    q = Queue(maxsize=(5 - 1) * 4)
    q.put(initial_pos)
    while not q.empty():
        x, y = q.get()
        moves = generate_legal_moves((x, y), board)
        # The cluster of chess pieces is not "surrounded", if one them can still move
        if len(moves) > 0:
            return []
        else:
            return_value += [(x, y)]
            marked[x][y] = player
            unchecked_nearby_allies = [
                (r, c)
                for (r, c) in NEIGHBORS_POSITIONS[x][y]
                if board[r][c] == player and marked[r][c] == 0
            ]
            for ally in unchecked_nearby_allies:
                q.put(ally)
    return return_value


def try_vay(old_pos, new_pos, board, player) -> List[Tuple]:
    """Tra ve danh sach cac vi tri quan doi phuong ma neu player di vao new_position thi co the vây/chẹt.
    board: before executing the move"""
    new_board = board_after_move(old_pos, new_pos, board, pure=True)
    x, y = new_pos
    # Get the list of opponent's chess pieces around the new position
    adjacent_enemies = [
        (r, c) for (r, c) in NEIGHBORS_POSITIONS[x][y] if board[r][c] == -player
    ]
    # The surrounded clusters may not be a singly contiguous region
    surrounded_enemies_by_regions = [
        contiguously_surrounded_pieces(pos, new_board) for pos in adjacent_enemies
    ]
    # Return the list of "vay/chet"-ed positions, without duplication
    return list(set(reduce(operator.add, surrounded_enemies_by_regions, [])))


def all_to_be_converted(src, des, board, player) -> List[Tuple]:
    """Return the list of opponent's chess pieces that current PLAYER can
    immediately convert by moving from SRC to DES."""
    return try_ganh(des, board, player) + try_vay(src, des, board, player)


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


TEST_BOARD_1 = [
    [-1, 00, +1, 00, +1],
    [00, -1, 00, 00, +1],
    [+1, +1, -1, 00, 00],
    [-1, -1, +1, +1, +1],
    [-1, 00, 00, -1, -1],
]


TEST_BOARD_2 = [
    [00, 00, +1, 00, +1],
    [-1, -1, 00, 00, +1],
    [-1, -1, +1, 00, 00],
    [-1, -1, +1, +1, +1],
    [-1, +1, 00, 00, +1],
]


def check_luat_mo(old_board, new_board, playing_player):
    """If "luật mở" is executing:
    Return the list of chess pieces which can move into the "mở" position and itself.
    Else return [] and None.
    -> tuple[list[tuple], tuple] | ([], None)
    """
    moved = infer_move(old_board, new_board)
    if moved is not None:
        src, des = moved
        # The list of positions which are converted by the opponent's previous move
        da_bi_an = [
            (i, j)
            for i in range(5)
            for j in range(5)
            if old_board[i][j] == playing_player and new_board[i][j] == -playing_player
        ]
        # Get all of our pieces which can move into the "mở" position
        possible_pieces_to_move = [
            pos
            for pos in NEIGHBORS_POSITIONS[src[0]][src[1]]
            if new_board[pos[0]][pos[1]] == playing_player
        ]
        if (
            len(da_bi_an) == 0
            and len(possible_pieces_to_move) > 0
            and len(try_ganh(src, new_board, playing_player)) > 0
        ):
            return (possible_pieces_to_move, src)
    return ([], None)


# Used to store the board's information after "our" previous move, to adhere to an "open move"
# Store 2 boards, to simulate 2 players playing with each other
previous_boards = {-1: deepcopy(INITIAL_BOARD), 1: deepcopy(INITIAL_BOARD)}


def move(board, player):
    # (board: List[List[int]], player: int)
    # -> Tuple[Tuple[int, int], Tuple[int, int]] | None
    # TODO: insert something useful here

    global previous_boards

    possible_pieces_to_move, open_position = check_luat_mo(
        old_board=previous_boards[player], new_board=board, playing_player=player
    )

    if open_position is not None and len(possible_pieces_to_move) > 0:
        # TODO: Decide which one to move
        src = possible_pieces_to_move[0]
        des = open_position
    else:

        # list[tuple[int,int]]
        owned_positions = [
            (r, c) for r in range(5) for c in range(5) if board[r][c] == player
        ]

        # list[tuple[tuple,tuple]]
        all_moves = [
            (pos, new_pos)
            for pos in owned_positions
            for new_pos in generate_legal_moves(pos, board)
        ]

        # dict[tuple[tuple,tuple]: list[tuple]]
        immediate_scores = {
            (src, des): all_to_be_converted(src, des, board, player)
            for (src, des) in all_moves
        }

        def get_immediate_score(move):
            return len(immediate_scores[move])

        max_immediate_score = max(map(get_immediate_score, all_moves))
        src, des = random.choice(
            [
                move
                for move in all_moves
                if get_immediate_score(move) == max_immediate_score
            ]
        )

    new_board = board_after_move_and_rules_application(src, des, board)

    previous_boards[player] = deepcopy(new_board)

    return (src, des)


def simulate():
    player = 1
    board = deepcopy(INITIAL_BOARD)
    while get_winner(board) == 0:
        src, des = move(deepcopy(board), player)
        board = board_after_move_and_rules_application(src, des, board)
        print(
            "Player ",
            player,
            " move from ",
            src,
            " to ",
            des,
            ", change the board to: ",
        )
        print(board_to_string(board))
        player = -player


# simulate()
