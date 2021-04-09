#!/usr/bin/env python3

import csv
import sys
import numpy as np
from dataclasses import dataclass
import random

# TODO: make it configurable
POPULATION_SIZE = 8

random_generator = np.random.default_rng()


@dataclass
class Grid:
    def __init__(self, grid, initial=None):
        self.current = grid
        # TODO: shared initial, N, max_score
        if initial is None:
            self.initial = np.copy(grid)
        else:
            self.initial = initial
        self.N = len(grid)
        # There are N^2 cells, each can take -1 penalty from row, column and sub grid contraints
        self.max_score = 3 * (self.N ** 2)


def main(args):
    grid = Grid(read_initial(args[1]))
    candidates = [
        Grid(gen_random_grid(grid.initial), grid.initial)
        for _ in range(POPULATION_SIZE)
    ]
    fitness_weights = [fitness(grid) for grid in candidates]
    while not (0 in fitness_weights):
        selected = random.choices(
            population=candidates, weights=fitness_weights, k=POPULATION_SIZE
        )
        fitness_weights = [fitness(grid) for grid in candidates]


def print_grid(grid: Grid):
    print()
    for i in range(grid.N):
        for k in range(grid.N):
            print(
                grid.current[i][k],
                " " if grid.initial[i][k] == 0 else ".",
                "\t",
                end="",
            )
        print()


def gen_random_grid(grid: list[list[int]]):
    N = len(grid)
    return [
        [(random_generator.integers(1, N + 1) if cell == 0 else cell) for cell in row]
        for row in grid
    ]


def read_initial(input_file):
    """Read initial numbers from input_file
    Return a list of cell lists; each cell, if initially declared, contains that number, else 0"""
    return [
        [(0 if cell == "" else int(cell)) for cell in row]
        for row in csv.reader(open(input_file))
    ]


def quick_test():
    main([None, "test.csv"])


def fitness(grid: Grid):
    rows = grid.current
    cols = np.transpose(rows)
    subs = get_sub_grids(rows)
    rows_fitness = sum(fitness_sub(row) for row in rows)
    cols_fitness = sum(fitness_sub(col) for col in cols)
    subs_fitness = sum(fitness_sub(sub) for sub in subs)
    return rows_fitness + cols_fitness + subs_fitness + grid.max_score


def fitness_sub(seq: list):
    """Return 0 - the number duplicate elements (excluding the first to be
    duplicated one) and 0s in seq."""
    uniq_elements = np.unique(seq)
    neg_repeated_amount = len(uniq_elements) - len(seq)
    if 0 in seq:
        return neg_repeated_amount - 1
    else:
        return neg_repeated_amount


def get_sub_grids(mat: list[list]) -> list[list]:
    N = len(mat)
    sqrt_N = int(np.sqrt(N))
    sub_grids_as_lists = [[] for _ in range(N)]
    for row in range(N):
        for col in range(N):
            sub_grids_as_lists[sub_grid_id(row, col, sqrt_N)].append(mat[row][col])
    return sub_grids_as_lists


def sub_grid_id(row, col, sqrt_N):
    """Return the appropriate ID of sub grid that cell at row, col would land in given N."""
    return (row // sqrt_N) * sqrt_N + col // sqrt_N


# quick_test()

if __name__ == "__main__":
    main(sys.argv)
