#!/usr/bin/env python3

import numpy as np
from functools import reduce
import operator
import math
import random

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
    return [(row, col) for (row, col) in unbounded_moves if (row >= 0 and col >= 0)]


def generate_legal_moves(position, board):
    moves = generate_all_moves(position)
    return [(position,(i, j)) for (i, j) in moves if board[i][j] == 0]

def generate_all_positions_moves(board, player):
    result = []
    for i in range(5):
        for j in range(5):
            if board[i][j] == player:
                result += generate_legal_moves((i,j))
    return result


def get_winner(board):
    pO_on_board = True in [(1 in row) for row in board]
    pX_on_board = True in [(-1 in row) for row in board]
    if pO_on_board and not pX_on_board:
        return 1
    if pX_on_board and not pO_on_board:
        return -1
    return 0


def board_after_move(source, dest, board):
    board[source[0]][source[1]], board[dest[0]][dest[1]] = (
        board[dest[0]][dest[1]],
        board[source[0]][source[1]],
    )
    return board

class Node:
    """ A node in the game tree. We wins if the viewpoint is    player 1
    """
    def __init__(self, player, move = None, parent = None, state = None):
        self.move = move
        self.parentNode = parent
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = generate_all_positions_moves(state, player) # future child nodes
        self.playerJustMoved = state.player # previous player

    def UCTChooseChild(self):
        """lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key = lambda c: c.wins/c.visits + math.sqrt(2*log(self.visits)/c.visits))[-1]
        return s
    
    def addChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(player= -self.playerJustMoved, move = m, parent = self, state = s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n
    
    def updateNode(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

def UCT(player, rootstate, iter, verbose = False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = Node(player = player,state = rootstate)

    for i in range(iter):
        node = rootnode
        state = rootstate[:]
        currentPlayer = -player  # turn of the player x

        # Select
        while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal, loop until have untried Child or it is leaf node
            # choose the next child
            node = node.UCTChooseChild()
            # update the state of the board ((x,y),(x,y))
            state[node.move[0][0]][node.move[0][1]] = 0
            state[node.move[1][0]][node.move[1][1]] = player
            currentPlayer = -currentPlayer

        # Expand
        if node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves) 
            # update the state of the board
            state[m[0][0]][m[0][1]] = 0
            state[m[1][0]][m[1][1]] = player
            currentPlayer = -currentPlayer
        
            node = node.addChild(m,state) # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        while get_winner(state) == 0: # while state is non-terminal
            choice = random.choice(generate_all_positions_moves(state, player))
            # update the state of the board
            state[choice[0][0]][choice[0][1]] = 0
            state[choice[1][0]][choice[1][1]] = player
            currentPlayer = -currentPlayer

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            winner = get_winner(state)
            if winner == -player:
                node.updateNode(1)
            else:
                node.updateNode(0)

            # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    return sorted(rootnode.childNodes, key = lambda c: c.visits)[-1].move # return the move that was most visited

def move(board, player):

    choice =  UCT(-player, board, 1000)
    if choice:
        return choice
    return None
