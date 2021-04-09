#!/usr/bin/env python3

import csv
import sys
import numpy as np
from dataclasses import dataclass


@dataclass
class Grid:
    def __init__(self, grid):
        self.current = grid
        self.initial = np.copy(grid)
        self.N = len(grid)


def main(args):
    grid = Grid(read_initial(args[1]))


def read_initial(input_file):
    """Read initial numbers from input_file
    Return a list of cell lists; each cell, if initially declared, is that number, else 0"""
    return [
        [(0 if cell == "" else int(cell)) for cell in row]
        for row in csv.reader(open(input_file))
    ]


def quick_test():
    main([None, "test.csv"])


def fitness(grid):
    """Max: 0"""
    rows = grid
    cols = np.transpose(grid)


def fitness_sub(seq):
    uniq_elements = np.unique(seq)
    repeated_number = len(seq) - len(uniq_elements)
    if 0 in seq:
        return repeated_number + 1
    else:
        return repeated_number


def get_sub_grids(grid):
    pass


if __name__ == "__main__":
    main(sys.argv)
