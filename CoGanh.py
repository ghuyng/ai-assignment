#!/usr/bin/env python3

import math
from math import log, inf
from operator import add as addop
import time, copy, random
from copy import deepcopy
from functools import reduce
from typing import List, Tuple

INITIAL_BOARD = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, -1],
    [-1, 0, 0, 0, -1],
    [-1, -1, -1, -1, -1],
]

# * PRIVATE


def _generate_neighbor_positions(position):
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
_NEIGHBORS_POSITIONS = [
    [_generate_neighbor_positions((i, j)) for j in range(5)] for i in range(5)
]


def _get_empty_neighbors(position, board) -> List[Tuple]:
    moves = _NEIGHBORS_POSITIONS[position[0]][position[1]]
    return [(i, j) for (i, j) in moves if board[i][j] == 0]


def _board_after_move_only(source, dest, board):
    """Return board after moving from source to dest without applying any capturing rules.
    Pure function."""
    new_board = deepcopy(board)
    x_old, y_old, x_new, y_new = source[0], source[1], dest[0], dest[1]
    new_board[x_new][y_new] = new_board[x_old][y_old]
    new_board[x_old][y_old] = 0
    return new_board


def _gen_opposite_position_pairs(position: Tuple[int, int]):
    """Return a list of pairs of opposing positions around the argument.
    -> List[Tuple[Tuple[int, int], Tuple[int, int]]]"""
    adjacent_positions = _NEIGHBORS_POSITIONS[position[0]][position[1]]
    x, y = position

    def helper(positions, return_pairs=[]):
        if len(positions) == 0:
            return return_pairs
        else:
            here = positions[0]
            other = (2 * x - here[0], 2 * y - here[1])
            return helper(
                [_ for _ in positions if not _ in (here, other)],
                return_pairs + ([(here, other)] if other in positions else []),
            )

    return helper(adjacent_positions)


# List of pairs of positions that the chess piece at [i][j] can potentially "gánh", regardless of "colors"
# List[List[ List[Tuple[Tuple,Tuple]] ]]
_OPPOSITE_POSITION_PAIRS = [
    [_gen_opposite_position_pairs((i, j)) for j in range(5)] for i in range(5)
]


def _get_possible_pairs_to_carry(pos, board, player):
    """Return the list of opponent's pieces that the PLAYER can gánh & vây/chẹt if they move from OLD_POS to NEW_POS.
    board: before executing the move"""
    pairs = _OPPOSITE_POSITION_PAIRS[pos[0]][pos[1]]
    return [
        ((r0, c0), (r1, c1))
        for ((r0, c0), (r1, c1)) in pairs
        if board[r0][c0] == board[r1][c1] == -player
    ]


def _try_carrying(src, des, board, player) -> List[Tuple]:
    """Return the list of opponent's pieces that the PLAYER can gánh & vây/chẹt if they move from OLD_POS to NEW_POS.
    board: before executing the move"""

    pairs = _OPPOSITE_POSITION_PAIRS[des[0]][des[1]]
    opponent = -player

    def get_pairs_to_be_captured(pair):
        x0, y0 = pair[0]
        x1, y1 = pair[1]
        if board[x0][y0] == board[x1][y1] == opponent:
            return [pair[0], pair[1]]
        return []

    captured_by_carrying = [
        pos for pair in pairs for pos in get_pairs_to_be_captured(pair)
    ]
    new_board = _board_after_move_only(src, des, board)
    for (r, c) in captured_by_carrying:
        new_board[r][c] = player
    # Surround after carrying
    captured_by_surrounding = _get_surrounded(new_board, -player)
    return captured_by_carrying + captured_by_surrounding


def _get_surrounded(board, player: int) -> List[Tuple]:
    """Return the list of positions of chess pieces of PLAYER on BOARD which are
    "vây/chẹt"-ed and can't move."""
    return_value = []
    # Save the visited positions to avoid repeated traversal
    marked = [[False for _ in range(5)] for _ in range(5)]
    # A stack for flood filling algorithm
    stk = []
    # A contiguous cluster of chess pieces that are surrounded unless one of them can move
    cluster = []
    for r in range(5):
        for c in range(5):
            if board[r][c] == player and not marked[r][c]:
                stk.append((r, c))
                # Whether current cluster is not surrounded
                free_cluster = False
                while len(stk) > 0:
                    x, y = stk.pop()
                    cluster += [(x, y)]
                    if not marked[x][y]:
                        marked[x][y] = True
                        moves = _get_empty_neighbors((x, y), board)
                        # If a piece in the current cluster can still move, mark the cluster as free
                        if len(moves) > 0:
                            free_cluster = True
                        # Process nearby unchecked allies
                        stk += [
                            (r, c)
                            for (r, c) in _NEIGHBORS_POSITIONS[x][y]
                            if board[r][c] == player and marked[r][c] == False
                        ]
                    # Add to the list of surrounded chess pieces
                if not free_cluster:
                    return_value += cluster
                cluster.clear()
    return return_value


def _try_surrounding(old_pos, new_pos, board, player) -> List[Tuple]:
    """Return the list of opponent's pieces that the PLAYER can vây/chẹt if they move from OLD_POS to NEW_POS.
    board: before executing the move"""
    new_board = _board_after_move_only(old_pos, new_pos, board)
    return _get_surrounded(new_board, -player)


def _infer_move(old_board, new_board):
    """Return the move which has been played, given old_board and new_board.
    Tuple -> [Tuple, Tuple] | None"""
    changed_positions = [
        (i, j) for i in range(5) for j in range(5) if old_board[i][j] != new_board[i][j]
    ]
    if changed_positions == []:
        return None
    else:
        # src position: became empty on the new board
        src = [pos for pos in changed_positions if new_board[pos[0]][pos[1]] == 0][0]
        # des position: was empty on the old board
        des = [pos for pos in changed_positions if old_board[pos[0]][pos[1]] == 0][0]
        return (src, des)


def _check_open_rule(old_board, new_board, player):
    """If "luật mở" is executing:
    Return the list of chess pieces which can move into the "mở" position and itself.
    Else return [] and None.
    -> tuple[list[tuple], tuple] | ([], None)
    """
    moved = _infer_move(old_board, new_board)
    if moved is not None:
        src, des = moved
        # The list of positions which are captured by the opponent's previous move
        captured = [
            (i, j)
            for i in range(5)
            for j in range(5)
            if old_board[i][j] == player and new_board[i][j] == -player
        ]
        # Get all of our pieces which can move into the "mở" position
        possible_pieces_to_move = [
            pos
            for pos in _NEIGHBORS_POSITIONS[src[0]][src[1]]
            if new_board[pos[0]][pos[1]] == player
        ]
        if (
            len(captured) == 0
            and len(possible_pieces_to_move) > 0
            and len(_get_possible_pairs_to_carry(src, new_board, player)) > 0
        ):
            return (possible_pieces_to_move, src)
    return ([], None)


def _rate_greedy_open(src, des, board, player) -> int:
    new_board = board_after_move_and_capturing(src, des, board)
    opponent_moves, is_opened = _check_open_rule(board, new_board, -player)

    def greedy_best_after_opp(opp_src, opp_des):
        captured_by_opponent = len(
            all_to_be_captured(opp_src, opp_des, new_board, -player)
        )
        new2_board = board_after_move_and_capturing(opp_src, opp_des, new_board)
        moves = get_all_legal_moves(new_board, new2_board, player)
        if len(moves) > 0:
            return max(
                len(all_to_be_captured(new2_src, new2_des, new2_board, player))
                - captured_by_opponent
                for (new2_src, new2_des) in moves
            )
        else:
            return 0

    if is_opened:
        return min(greedy_best_after_opp(opp_src, src) for opp_src in opponent_moves)
    else:
        return 0


def _greedy_alg(prev_board, board, player):
    # Dumb Greedy algorithm: randomly choose from the moves which has the best immediate reward
    # dict[tuple[tuple,tuple]: list[tuple]]
    moves = get_all_legal_moves(prev_board, board, player)
    if len(moves) > 0:
        immediate_scores = {
            (src, des): all_to_be_captured(src, des, board, player)
            for (src, des) in moves
        }

        def get_immediate_score(move):
            l = len(immediate_scores[move])
            return l if l > 0 else _rate_greedy_open(move[0], move[1], board, player)

        max_immediate_score = max(map(get_immediate_score, moves))
        src, des = random.choice(
            [move for move in moves if get_immediate_score(move) == max_immediate_score]
        )
        return (src, des)
    else:
        return None


# * PUBLIC

# Used to store the board's information after "our" previous move, to adhere to an "open move"
# Store 2 boards, to simulate 2 players playing with each other
prev_boards = {i: deepcopy(INITIAL_BOARD) for i in [-1, 1]}


def benchmark(func, *args):
    time0 = time.time()
    val = func(*args)
    time1 = time.time()
    print(func, args, f"takes {time1 - time0}s")
    return val


def board_after_move_and_capturing(src, des, board):
    """Return board after moving from src to des and apply capturing rules.
    Pure function."""
    player = board[src[0]][src[1]]
    capturing = all_to_be_captured(src, des, board, player)
    new_board = _board_after_move_only(src, des, board)
    for (r, c) in capturing:
        new_board[r][c] = player
    return new_board


def get_winner(board) -> int:
    pO_on_board = True in [(1 in row) for row in board]
    pX_on_board = True in [(-1 in row) for row in board]
    if pO_on_board and not pX_on_board:
        return 1
    if pX_on_board and not pO_on_board:
        return -1
    return 0


def board_to_string(board) -> str:
    return reduce(
        addop,
        [("\t " if i > 0 else "\n") + str(row[i]) for row in board for i in range(5)],
    )


def print_board(b):
    print(board_to_string(b))


def all_to_be_captured(src, des, board, capturer) -> List[Tuple]:
    """Return the list of opponent's chess pieces that current PLAYER can
    immediately capture by moving from SRC to DES."""
    # Avoid duplication
    return list(
        set(
            _try_carrying(src, des, board, capturer)
            + _try_surrounding(src, des, board, capturer)
        )
    )


def get_all_legal_moves(
    old_board, current_board, player: int
) -> List[Tuple[Tuple, Tuple]]:
    """Return the list of all possible moves according to rules."""
    possible_pieces_to_move, open_position = _check_open_rule(
        old_board=old_board, new_board=current_board, player=player
    )
    if open_position and len(possible_pieces_to_move) > 0:
        return [(pos, open_position) for pos in possible_pieces_to_move]
    else:
        # list[tuple[int,int]]
        owned_positions = [
            (r, c) for r in range(5) for c in range(5) if current_board[r][c] == player
        ]
        # list[tuple[tuple,tuple]]
        return [
            (pos, new_pos)
            for pos in owned_positions
            for new_pos in _get_empty_neighbors(pos, current_board)
        ]


def rate_board(board, player):
    """Return the difference of PLAYER and OPPONENT's number possible moves on
    BOARD, approximately."""
    approx_player_moves = sum(
        len(_get_empty_neighbors((i, j), board))
        for i in range(5)
        for j in range(5)
        if board[i][j] == player
    )
    approx_opponent_moves = sum(
        len(_get_empty_neighbors((i, j), board))
        for i in range(5)
        for j in range(5)
        if board[i][j] == -player
    )
    return approx_player_moves - approx_opponent_moves


def minimax_rate_move(src, des, board, player, height):
    # The board after moving
    new_board = board_after_move_and_capturing(src, des, board)
    opponent_moves = get_all_legal_moves(board, new_board, -player)
    # Terminal node
    if height <= 0 or len(opponent_moves) == 0:
        return rate_board(new_board, player)
    # Recursively apply the algorithm on opponent's moves
    else:
        return max(
            -minimax_rate_move(_src, _des, new_board, -player, height - 1)
            for (_src, _des) in opponent_moves
        )


def minimax_alg(prev_board, board, player, initial_height):
    moves = get_all_legal_moves(prev_board, board, player)
    if len(moves) > 0:
        scores = {
            (src, des): minimax_rate_move(src, des, board, player, initial_height)
            for (src, des) in moves
        }

        def get_score(move):
            return scores[move]

        max_score = max(map(get_score, moves))
        src, des = random.choice(
            [move for move in moves if get_score(move) == max_score]
        )
        return (src, des)
    else:
        return None


def ab_rate_board(
    prev_board, board, previous_player, original_player, height, al=-inf, be=inf
) -> float:
    is_maximizing_player = -previous_player == original_player
    next_moves = get_all_legal_moves(prev_board, board, -previous_player)
    if height == 0 or len(next_moves) == 0:
        return rate_board(board, original_player)
    if is_maximizing_player:
        value = -inf
        for (src, des) in next_moves:
            new_board = board_after_move_and_capturing(src, des, board)
            new_val = ab_rate_board(
                board, new_board, -previous_player, original_player, height - 1, al, be
            )
            value = max(value, new_val)
            al = max(al, value)
            if value >= be:
                break
    else:
        value = inf
        for (src, des) in next_moves:
            new_board = board_after_move_and_capturing(src, des, board)
            new_val = ab_rate_board(
                board, new_board, -previous_player, original_player, height - 1, al, be
            )
            value = min(value, new_val)
            be = min(be, value)
            if value <= al:
                break
    return value


def ab_pruning_alg(prev_board, board, player, initial_height):
    moves = get_all_legal_moves(prev_board, board, player)
    if len(moves) > 0:
        # Construct a dictionary of move : rated score
        scores = {
            (src, des): ab_rate_board(
                board,
                board_after_move_and_capturing(src, des, board),
                player,
                player,
                initial_height,
            )
            for (src, des) in moves
        }

        def get_score(move):
            return scores[move]

        max_score = max(map(get_score, moves))
        # print(scores)
        src, des = random.choice(
            [move for move in moves if get_score(move) == max_score]
        )
        return (src, des)
    else:
        return None


class Node:
    """A node in the game tree. We wins if the viewpoint is    player 1"""

    def __init__(self, player, move=None, parent=None, state=None, preState=None):
        self.move = move
        self.parentNode = parent
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = get_all_legal_moves(
            preState, state, -player
        )  # future child nodes
        self.playerJustMoved = player  # previous player

    def UCTChooseChild(self):
        """lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
        exploration versus exploitation.
        """
        s = sorted(
            self.childNodes,
            key=lambda c: c.wins / c.visits
            + math.sqrt(2 * log(self.visits) / c.visits),
        )[-1]
        return s

    def addChild(self, m, s, preS):
        """Remove m from untriedMoves and add a new child node for this move.
        Return the added child node
        """
        n = Node(
            player=-self.playerJustMoved, move=m, parent=self, state=s, preState=preS
        )
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n

    def updateNode(self, result):
        """Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved."""
        self.visits += 1
        self.wins += result


def UCT(player, rootstate, iter, prev_board, verbose=False):
    """Conduct a UCT search for itermax iterations starting from rootstate.
    Return the best move from the rootstate.
    Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""
    rootnode = Node(player=player, state=rootstate, preState=prev_board)

    for i in range(iter):
        node = rootnode
        state = deepcopy(rootstate)
        pre_state = deepcopy(state)
        currentPlayer = -player  # turn of the player x

        # Select
        while (
            node.untriedMoves == [] and node.childNodes != []
        ):  # node is fully expanded and non-terminal, loop until have untried Child or it is leaf node
            # choose the next child
            node = node.UCTChooseChild()
            # update the state of the board ((x,y),(x,y))
            pre_state = deepcopy(state)
            state = board_after_move_and_capturing(node.move[0], node.move[1], state)
            currentPlayer = -currentPlayer

        # Expand
        if (
            node.untriedMoves != []
        ):  # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves)
            # update the state of the board
            pre_state = deepcopy(state)
            state = board_after_move_and_capturing(m[0], m[1], state)
            currentPlayer = -currentPlayer

            node = node.addChild(m, state, pre_state)  # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        height = 0
        while get_winner(state) == 0:  # while state is non-terminal
            if height > 30:
                break
            choice = random.choice(get_all_legal_moves(pre_state, state, currentPlayer))
            # update the state of the board
            pre_state = deepcopy(state)
            state = board_after_move_and_capturing(choice[0], choice[1], state)
            # print_board(state)
            currentPlayer = -currentPlayer
            height += 1

        # cout board
        # print(f"The board becomes {board_to_string(state)}")

        # Backpropagate
        while (
            node != None
        ):  # backpropagate from the expanded node and work back to the root node
            if height > 30:
                winner = player
            else:
                winner = get_winner(state)

            if winner == -player:
                node.updateNode(1)
            else:
                node.updateNode(0)

            # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    # cout root node
    # for i in rootnode.childNodes:
    #     print(i.move)
    #     print(i.wins)
    #     print(i.visits)

    return sorted(rootnode.childNodes, key=lambda c: c.visits)[
        -1
    ].move  # return the move that was most visited


def move(board, player):
    # (board: List[List[int]], player: int)
    # -> Tuple[Tuple[int, int], Tuple[int, int]] | None
    move = choose_move_alg1(prev_boards[player], board, player)
    if not move:
        return None
    src, des = move
    new_board = board_after_move_and_capturing(src, des, board)
    prev_boards[player] = deepcopy(new_board)

    return (src, des)


def simulate():
    player = 1
    board = deepcopy(INITIAL_BOARD)
    turn = 0
    while get_winner(board) == 0:
        time_before = time.time()
        if player == -1:
            src, des = choose_move_alg0(prev_boards[player], deepcopy(board), player)
        else:
            src, des = choose_move_alg1(prev_boards[player], deepcopy(board), player)
        time_after = time.time()
        prev_boards[player] = deepcopy(board)

        board = board_after_move_and_capturing(src, des, board)
        turn += 1
        print(
            f"Turn {turn}: {player} moved from {src} to {des} after {time_after - time_before}s, the board became: {board_to_string(board)}"
        )
        player = -player
    print(f"Player {get_winner(board)} won.")


# * EDIT THIS!


def choose_move_alg0(prev_board, board, player):
    # return minimax_alg(prev_board, board, player, initial_height=1)
    # return _greedy_alg(prev_board, board, player)
    return ab_pruning_alg(prev_board, board, player, initial_height=3)


def choose_move_alg1(prev_board, board, player):
    # return minimax_alg(prev_board, board, player, initial_height=1)
    return ab_pruning_alg(prev_board, board, player, initial_height=3)
    # return UCT(player=player, rootstate=board, iter=512, prev_board=prev_board)
    # return choose_move_alg0(prev_board, board, player)


# simulate()
