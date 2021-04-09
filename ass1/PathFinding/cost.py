import math
# Manhattan Distance => suitable for moving only in four directions only (right, left, top, bottom)
def Manhattan_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

# Diagonal distance => suitable for moving in eight directions only (similar to a move of a King in Chess)
def Diagonal_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return max(abs(x1 - x2), abs(y1 - y2))

# Euclidean distance => suitable for moving in any direction

def Euclidean_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
