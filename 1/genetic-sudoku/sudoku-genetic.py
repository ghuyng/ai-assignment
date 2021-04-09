#!/usr/bin/env python3

import csv
import sys
import numpy as np
from dataclasses import dataclass
import random

# TODO: make it configurable
POPULATION_SIZE = 1024

random_generator = np.random.default_rng()


@dataclass
class Grid:
    def __init__(self, grid: list[list[int]], initial=None):
        self.current = np.copy(grid)
        # TODO: shared initial, N, max_score
        if initial is None:
            self.initial = np.copy(grid)
        else:
            self.initial = initial
        self.N = len(grid)
        # There are N^2 cells, each can take -1 penalty from : row, column and sub grid constraints
        self.max_score = 3 * (self.N ** 2)
        self.sqrtN = int(np.sqrt(self.N))


def main(args):
    # Read the partially filled grid from file
    grid = Grid(read_initial(args[1]))
    # Generate initial population
    candidates = [
        Grid(gen_random_grid(grid.initial), grid.initial)
        for _ in range(POPULATION_SIZE)
    ]
    # Weight each candidates
    fitness_weights = list(map(fitness, candidates))
    # Record current generation's NO.
    generation_no = 0
    # Loop until a perfect solution is found
    while not (grid.max_score in fitness_weights):

        # Print info
        print("\n\nGeneration: ", generation_no)
        for i in range(POPULATION_SIZE):
            print_grid(candidates[i])
            print("Score: ", fitness_weights[i])

        # Randomly select candidates by fitness weights
        selected = random.choices(
            population=candidates, weights=fitness_weights, k=POPULATION_SIZE
        )
        # Breed selected candidates
        children = crossover(selected)
        # Mutate the children and have them replace candidates
        candidates = list(map(mutate, children))
        # Re-calculate fitness weights
        fitness_weights = list(map(fitness, candidates))
        generation_no += 1

    # Get the perfect solution and display it
    solution = candidates[fitness_weights.index(grid.max_score)]
    print_grid(solution)


def crossover(grids: list[Grid], children: list[Grid] = []) -> list[Grid]:
    """Perform "crossover" step for each pair in grids.
    Return the accumulated children."""
    if grids == []:
        return children
    else:
        N = grids[0].N
        # Identify the line to crossover
        point = random_generator.integers(N - 1)
        # Crossover: split at `point`, swap [0,point) and [point,N)
        child0 = Grid(
            np.concatenate((grids[0].current[0:point], grids[1].current[point:N])),
            grids[0].initial,
        )
        child1 = Grid(
            np.concatenate((grids[1].current[0:point], grids[0].current[point:N])),
            grids[0].initial,
        )
        return crossover(grids[2:], children + [child0, child1])


def mutate(grid: Grid) -> Grid:
    """Perform "mutate" on grid."""
    # Randomly select a row and column
    row = random_generator.integers(grid.N)
    col = random_generator.integers(grid.N)
    # Replace with a random value unless the cell is immutable
    if grid.initial[row][col] == 0:
        # sub_grid_values = get_sub_grids(grid.current)[sub_grid_id(row, col, grid.sqrtN)]
        # filled_values = set(
        # grid.current[row] + np.transpose(grid.current)[col] + sub_grid_values
        # )
        # unfilled_values = list(set(range(grid.N)) - filled_values)
        # if unfilled_values == []:
        # grid.current[row][col] = random_generator.integers(1, grid.N + 1)
        # else:
        # grid.current[row][col] = random_generator.choice(unfilled_values)
        grid.current[row][col] = random_generator.integers(1, grid.N + 1)
    return grid


def print_grid(grid: Grid):
    """Pretty(?) print grid."""
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
    """Return a random grid from the argument by randomly fill all mutable cells."""
    N = len(grid)
    return [
        [(random_generator.integers(1, N + 1) if cell == 0 else cell) for cell in row]
        for row in grid
    ]


def read_initial(input_file):
    """Read initial numbers from input_file
    Return a list of cell lists; each cell, if initially declared, contains that number, else 0."""
    return [
        [(0 if cell == "" else int(cell)) for cell in row]
        for row in csv.reader(open(input_file))
    ]


def quick_test():
    """For REPL interaction."""
    main([None, "test.csv"])


def fitness(grid: Grid):
    """Determine grid's fitness score."""
    rows = grid.current
    cols = np.transpose(rows)
    subs = get_sub_grids(rows)
    # These three are negative
    rows_fitness = sum(fitness_sub(row) for row in rows)
    cols_fitness = sum(fitness_sub(col) for col in cols)
    subs_fitness = sum(fitness_sub(sub) for sub in subs)
    # Ensure positiveness by adding with the maximum score
    return rows_fitness + cols_fitness + subs_fitness + grid.max_score


def fitness_sub(seq: list):
    """Return 0 - (negative) the number duplicate elements (excluding the first to be
    duplicated one) and 0s in seq.
    Since 0 cells are unfilled cells."""
    uniq_elements = np.unique(seq)
    neg_repeated_amount = len(uniq_elements) - len(seq)
    if 0 in seq:
        return neg_repeated_amount - 1
    else:
        return neg_repeated_amount


def get_sub_grids(mat: list[list[int]]) -> list[list[int]]:
    """Return N sub squared grids from NxN grid."""
    N = len(mat)
    sqrt_N = int(np.sqrt(N))
    sub_grids_as_lists = [[] for _ in range(N)]
    for row in range(N):
        for col in range(N):
            sub_grids_as_lists[sub_grid_id(row, col, sqrt_N)].append(mat[row][col])
    return sub_grids_as_lists


def sub_grid_id(row, col, sqrt_N):
    """Return the appropriate ID of sub grid that cell at row, col would land, given N.
    For example:
0 	0 	0 	1 	1 	1 	2 	2 	2
0 	0 	0 	1 	1 	1 	2 	2 	2
    ...
3 	3 	3 	4 	4 	4 	5 	5 	5
    ...
6 	6 	6 	7 	7 	7 	8 	8 	8 	"""
    return (row // sqrt_N) * sqrt_N + col // sqrt_N


# quick_test()

if __name__ == "__main__":
    main(sys.argv)
