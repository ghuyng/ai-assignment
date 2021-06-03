#!/usr/bin/env python3

import numpy as np
from functools import reduce
import operator

import time, copy, random

from typing import List, Tuple

# from queue import Queue

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


def print_board(b):
    print(board_to_string(b))


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


def board_after_move(source, dest, board):
    """Return board after moving from source to dest without applying any converting rules.
    Pure function."""
    new_board = deepcopy(board)
    x_old, y_old, x_new, y_new = source[0], source[1], dest[0], dest[1]
    new_board[x_new][y_new] = new_board[x_old][y_old]
    new_board[x_old][y_old] = 0
    return new_board


def board_after_move_and_rules_application(src, des, board):
    """Return board after moving from src to des and apply converting rules.
    Pure function."""
    player = board[src[0]][src[1]]
    converting = all_to_be_converted(src, des, board, player)
    new_board = board_after_move(src, des, board)
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


def get_possible_couples_to_carry(pos, board, player):
    """Tra ve danh sach cac vi tri quan doi phuong ma neu player di vao new_position thi co the ganh.
    board: before executing the move"""
    pairs = OPPOSITE_POSITION_PAIRS[pos[0]][pos[1]]
    return [
        ((r0, c0), (r1, c1))
        for ((r0, c0), (r1, c1)) in pairs
        if board[r0][c0] == board[r1][c1] == -player
    ]


def try_ganh(src, des, board, player) -> List[Tuple]:
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

    pairs = OPPOSITE_POSITION_PAIRS[des[0]][des[1]]
    opponent = -player

    def cap_quan_bi_ganh(pair):
        x0, y0 = pair[0]
        x1, y1 = pair[1]
        if board[x0][y0] == board[x1][y1] == opponent:
            return [pair[0], pair[1]]
        return []

    converted_by_ganh = [pos for pair in pairs for pos in cap_quan_bi_ganh(pair)]
    new_board = board_after_move(src, des, board)
    for (r, c) in converted_by_ganh:
        new_board[r][c] = player
    # Surround after carrying
    converted_by_surrounding = get_surrounded(new_board, -player)
    return converted_by_ganh + converted_by_surrounding


def get_surrounded(board, player: int) -> List[Tuple]:
    return_value = []
    # Save the visited positions to avoid repeated traversal
    marked = [[False for _ in range(5)] for _ in range(5)]
    for r in range(5):
        for c in range(5):
            if board[r][c] == player:
                # A stack for flood filling algorithm
                stk = [(r, c)]
                # A contiguous cluster of chess pieces that are surrounded unless one of them can move
                cluster = []
                # Whether current cluster is not surrounded
                free_cluster = False
                while len(stk) > 0:
                    x, y = stk.pop()
                    # If the position has already been processed, ignore it
                    if not marked[x][y]:
                        # Don't append to the cluster before checking marked,
                        # because the position may have already been marked, and
                        # belongs to a free cluster
                        cluster += [(x, y)]
                        marked[x][y] = True
                        moves = generate_legal_moves((x, y), board)
                        # If a piece in the current cluster can still move, mark the cluster as free
                        if len(moves) > 0:
                            free_cluster = True
                        # Process nearby unchecked allies
                        for ally in [
                            (r, c)
                            for (r, c) in NEIGHBORS_POSITIONS[x][y]
                            if board[r][c] == player and marked[r][c] == False
                        ]:
                            stk.append(ally)
                    # Add to the list of surrounded chess pieces
                if not free_cluster:
                    return_value += cluster
    return return_value


def try_vay(old_pos, new_pos, board, player) -> List[Tuple]:
    """Tra ve danh sach cac vi tri quan doi phuong ma neu player di vao new_position thi co the vây/chẹt.
    board: before executing the move"""
    new_board = board_after_move(old_pos, new_pos, board)
    x, y = new_pos
    # Get the list of opponent's chess pieces around the new position
    # adjacent_enemies = [
    #     (r, c) for (r, c) in NEIGHBORS_POSITIONS[x][y] if board[r][c] == -player
    # ]
    return get_surrounded(new_board, -player)


def all_to_be_converted(src, des, board, player) -> List[Tuple]:
    """Return the list of opponent's chess pieces that current PLAYER can
    immediately convert by moving from SRC to DES."""
    return try_ganh(src, des, board, player) + try_vay(src, des, board, player)


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
            and len(get_possible_couples_to_carry(src, new_board, playing_player)) > 0
        ):
            return (possible_pieces_to_move, src)
    return ([], None)


# Used to store the board's information after "our" previous move, to adhere to an "open move"
# Store 2 boards, to simulate 2 players playing with each other
previous_boards = {-1: deepcopy(INITIAL_BOARD), 1: deepcopy(INITIAL_BOARD)}


def move(board, player):
    # (board: List[List[int]], player: int)
    # -> Tuple[Tuple[int, int], Tuple[int, int]] | None

    global previous_boards

    possible_pieces_to_move, open_position = check_luat_mo(
        old_board=previous_boards[player], new_board=board, playing_player=player
    )

    if open_position is not None and len(possible_pieces_to_move) > 0:
        src, des = choose_move(
            board, player, [(pos, open_position) for pos in possible_pieces_to_move]
        )
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
        src, des = choose_move(board, player, all_moves)

    new_board = board_after_move_and_rules_application(src, des, board)

    previous_boards[player] = deepcopy(new_board)

    return (src, des)


def choose_move(board, player, moves: list[Tuple[Tuple, Tuple]]) -> Tuple[Tuple, Tuple]:
    """TODO: Implement this!!!"""
    # Dumb Greedy algorithm: randomly choose from the moves which has the best immediate reward
    # dict[tuple[tuple,tuple]: list[tuple]]
    immediate_scores = {
        (src, des): all_to_be_converted(src, des, board, player) for (src, des) in moves
    }

    def get_immediate_score(move):
        return len(immediate_scores[move])

    max_immediate_score = max(map(get_immediate_score, moves))
    src, des = random.choice(
        [move for move in moves if get_immediate_score(move) == max_immediate_score]
    )
    return (src, des)


def simulate():
    player = 1
    board = deepcopy(INITIAL_BOARD)
    turn = 0
    while get_winner(board) == 0:
        src, des = move(deepcopy(board), player)
        board = board_after_move_and_rules_application(src, des, board)
        turn += 1
        print(
            "Turn: ",
            turn,
            "Player",
            player,
            "moves from",
            src,
            "to",
            des,
            ", change the board to:",
        )
        print_board(board)
        player = -player


# simulate()
