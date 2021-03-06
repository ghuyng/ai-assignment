# N puzzle with Depth First Search (N=k*k-1) matrix k*k

# Start state: 
# - [i(1),i(2),i(3),...,i(k-2),i(k-1)]
# - i(k) in range [0,k-1]
# - i(u) differs i(v) (u differs v) with u,v in range [0,k-1]

# Rules to move: suppose that we move value 0
# -  Move left: [i(0),...,i(n-2),a,0,i(n+1),...,i(k-1)] -> [i(1),...,i(n-2),0,a,i(n+1),...,i(k-1)] (if 0 is not in the first column)
# -  Move right: [i(0),...,i(n-2),0,a,i(n+1),...,i(k-1)] -> [i(1),...,i(n-2),a,0,i(n+1),...,i(k-1)] (if 0 is not in the last column)
# -  Move up: [i(0),...,i(n-k),a,...,i(n),0,...,i(k-1)] -> [i(1),...,i(n-k),0,...,i(n),0,...,i(k-1)] (if 0 is not in the first row)
# -  Move down: [i(0),...,i(n),0,...,i(n+k),a,...,i(k-1)] -> [i(1),...,i(n),a,...,i(n+k),0,...,i(k-1)] (if 0 is not in the last row)

# End state: [0,1,2,3,...,k-2,k-1]
#       0 | 1 | 2
#       3 | 4 | 5
#       6 | 7 | 8

from numpy import random
import argparse

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

goalStringState = ""
N = 0

# Class node store a state of the puzzle with some information
# - state: a state of the puzzle
# - parent: the previous state of this state
# - depth: number of moves need to go to this state
# - k: puzzle k*k-1
class Node:
    def __init__(self, state, parent, depth):
        self.state = state
        self.parent = parent
        self.depth = depth
        if self.state:
            self.k = int(len(state) ** 0.5)
            self.stringState = "".join(str(i) for i in self.state)

    def showState(self):
        print("Step " + str(self.depth) + ":")
        # print out the State
        for i in range(len(self.state)):
            print(self.state[i], "|", end="")
            if (i + 1) % self.k == 0:
                print()
        for i in range(self.k):
            print("--", end="")
        print()


# function generateInitState: create a goal state and also start state (if users don't make init state themselves)
# - k: size of puzzle (k*k)
# - init: if None, function will create a start state
# - isRand: true if user wants to create a start state
# Return: the node contains init state
def generateInitState(k, init=None, isRand=True):
    global goalStringState

    a = [i for i in range(0, k * k)]
    # assign End State
    goalStringState = "".join(str(i) for i in a)

    if isRand:
        # create a random input
        for i in range(0, 20):
            index1 = random.randint(0, k * k - 1)
            index2 = random.randint(0, k * k - 1)
            tmp = a[index1]
            a[index1] = a[index2]
            a[index2] = tmp
    else:
        a = init

    node = Node(a, None, 0)
    node.showState()
    print("******************")
    return node


# Function dfs
# - initState: the input state (start state)
# - return : End state (the goal state)
def dfs(initNode):
    global goalStringState

    visitedStateSet = set()  # list string of visited states
    stack = list([initNode])

    while stack:
        node = stack.pop()

        if node.stringState in visitedStateSet:
            continue

        visitedStateSet.add(node.stringState)

        if node.stringState == goalStringState:
            return node

        childNodes = findChildNodes(node)

        for child in childNodes:
            if child.stringState not in visitedStateSet:
                stack.append(child)

    return None

# function: findChildNodes: search childNode
# - node: the previous node
# return: a list that contains child's nodes
def findChildNodes(node):
    childList = []

    childList.append(Node(slide(RIGHT, node), node, node.depth + 1))
    childList.append(Node(slide(DOWN, node), node, node.depth + 1))
    childList.append(Node(slide(LEFT, node), node, node.depth + 1))
    childList.append(Node(slide(UP, node), node, node.depth + 1))

    # remove None Node
    childList = [child for child in childList if child.state]

    return childList

# function slide: move the value 0
# - position: UP, DOWN, LEFT, RIGHT
# - parent: the previous node
# return: an array that contains new state
def slide(position, parent):
    childState = parent.state[:]
    # find index of 0 in parent state
    index = childState.index(0)
    # size of one dimension of the puzzle
    k = parent.k

    if position == UP:
        # can only slide 0 UP if 0 is not in the first line of puzzle
        if index >= k:
            # swap 0 UP
            swap(childState, index, index - k)
            return childState
        else:
            return None
    elif position == DOWN:
        # can only slide 0 DOWN if 0 is not in the final line of puzzle
        if index < k * k - k:
            # swap 0 DOWN
            swap(childState, index, index + k)
            return childState
        else:
            return None
    elif position == LEFT:
        # can only slide 0 LEFT if 0 is not in the first column of puzzle
        if index not in range(0, k * k, k):
            # swap 0 DOWN
            swap(childState, index, index - 1)
            return childState
        else:
            return None
    else:
        # can only slide 0 RIGHT if 0 is not in the final column of puzzle
        if index not in range(k - 1, k * k, k):
            # swap 0 RIGHT
            swap(childState, index, index + 1)
            return childState
        else:
            return None

# function swap: change the position of value 0
# - state: an array need to be change
# - index0: index of value 0
# - destIndex: index of value 0 's new position
def swap(state, index0, destIndex):
    state[index0] = state[destIndex]
    state[destIndex] = 0


# Print the solution step by step from the final state down to init state
def showSolution(finalnode):
    if finalnode:
        depth = finalnode.depth
        curNode = finalnode
        while curNode:
            curNode.showState()
            curNode = curNode.parent
        print("Number of steps:" + str(depth))
    else:
        print("No solution")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k",
        default=2,
        type=int,
        help="This is k-puzzle",
    )
    parser.add_argument(
        "--initArr",
        default=[],
        nargs="+",
        type=int,
        help="Array of integers of start state",
    )
    args = parser.parse_args()
    if args.initArr:
        startState = generateInitState(args.k, args.initArr, False)
    else:
        startState = generateInitState(args.k)

    resultState = dfs(startState)
    showSolution(resultState)


if __name__ == "__main__":
    main()
