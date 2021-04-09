#!/usr/bin/env python3

import csv
import sys
import numpy as np
from dataclasses import dataclass

# TODO: make it configurable
POPULATION_SIZE = 8

random_generator = np.random.default_rng()


@dataclass
class Grid:
    def __init__(self, grid, initial=None):
        self.current = grid
        # TODO: shared initial
        if initial is None:
            self.initial = np.copy(grid)
        else:
            self.initial = initial
        self.N = len(grid)


def main(args):
    grid = Grid(read_initial(args[1]))
    candidates = [
        Grid(gen_random_grid(grid.initial), grid.initial)
        for _ in range(POPULATION_SIZE)
    ]
    for g in candidates:
        print_grid(g)


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
    """Max: 0"""
    rows = grid.current
    cols = np.transpose(rows)
    subs = get_sub_grids(rows)
    rows_fitness = sum(fitness_sub(row) for row in rows)
    cols_fitness = sum(fitness_sub(col) for col in cols)
    subs_fitness = sum(fitness_sub(sub) for sub in subs)
    return rows_fitness + cols_fitness + subs_fitness


def fitness_sub(seq: list):
    uniq_elements = np.unique(seq)
    repeated_amount = len(uniq_elements) - len(seq)
    if 0 in seq:
        return repeated_amount - 1
    else:
        return repeated_amount


def get_sub_grids(mat: list[list]) -> list[list]:
    N = len(mat)
    sqrt_root_N = int(np.sqrt(N))
    sub_grids_as_lists = [[] for _ in range(N)]
    for row in range(N):
        for col in range(N):
            sub_grids_as_lists[sub_grid_id(row, col, sqrt_root_N)].append(mat[row][col])
    return sub_grids_as_lists


def sub_grid_id(row, col, sqrt_root_N):
    return (row // sqrt_root_N) * sqrt_root_N + col // sqrt_root_N


# quick_test()

if __name__ == "__main__":
    main(sys.argv)
