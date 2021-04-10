#!/usr/bin/env python3

from functools import reduce
import csv
import sys
import numpy as np
from dataclasses import dataclass
import random
import operator

random_generator = np.random.default_rng()


@dataclass
class Grid:
    data: list[list[int]]

    def to_string(self, init: list[list[int]]):
        N = len(self.data)
        rows_to_strs = [
            reduce(
                operator.add,
                [
                    str(self.data[row][col])
                    + (" " if init[row][col] == 0 else ".")
                    + "\t"
                    for col in range(N)
                ],
            )
            + "\n"
            for row in range(N)
        ]
        return reduce(operator.add, rows_to_strs)


@dataclass
class Population:
    def __init__(self, initial: list[list[int]], population_size: int):
        self.initial = Grid(initial)
        self.N = len(initial)
        self.sqrtN = int(np.sqrt(self.N))
        # There are N^2 cells, each can take -1 penalty from : row, column and sub grid constraints
        self.max_score = 3 * (self.N ** 2)
        self.population_size = population_size
        # Generate initial population
        self.candidates = [
            Grid(gen_random_grid(self.initial)) for _ in range(self.population_size)
        ]


def main(args):
    # Read the partially filled grid from file
    population = Population(read_initial(args[1]), int(args[2]))
    # Weight each candidates
    fitness_weights = list(map(fitness, population.candidates))
    # Record current generation's NO.
    generation_no = 0
    # Loop until a perfect solution is found
    while not (population.max_score in fitness_weights):

        # Print info
        print("\n\nGeneration: ", generation_no)
        for i in range(population.population_size):
            print(population.candidates[i].to_string(population.initial.data))
            print("Score: ", fitness_weights[i])

        # Randomly select candidates by fitness weights
        selected = random.choices(
            population=population.candidates,
            weights=fitness_weights,
            k=population.population_size,
        )
        # Breed selected candidates
        children = crossover(selected)
        # Mutate the children and have them replace candidates
        population.candidates = [
            mutate(child, population.initial) for child in children
        ]
        # Re-calculate fitness weights
        fitness_weights = list(map(fitness, population.candidates))
        generation_no += 1

    # Get the perfect solution and display it
    solution = population.candidates[fitness_weights.index(population.max_score)]
    print(solution.to_string(population.initial.data))


def crossover(candidates: list[Grid], children: list[Grid] = []) -> list[Grid]:
    """Perform "crossover" step for each pair in candidates.
    Return the accumulated children."""
    if candidates == []:
        return children
    else:
        N = len(candidates[0].data)
        # Identify the line to crossover
        point = random_generator.integers(1, N - 1)
        # Crossover: split at `point`, swap [0,point) and [point,N)
        # normal list concatenation does not work on numpy matrices
        child0 = Grid(
            np.concatenate((candidates[0].data[0:point], candidates[1].data[point:N]))
        )
        child1 = Grid(
            np.concatenate((candidates[1].data[0:point], candidates[0].data[point:N]))
        )
        return crossover(candidates[2:], children + [child0, child1])


def mutate(grid: Grid, initial: Grid) -> Grid:
    """Perform "mutate" on grid, leave initial cells intact."""
    # Randomly select a row and column
    row = random_generator.integers(len(grid.data))
    col = random_generator.integers(len(grid.data))
    # Replace with a random value unless the cell is immutable
    if initial.data[row][col] == 0:
        grid.data[row][col] = random_generator.integers(1, len(initial.data) + 1)
    return grid


def gen_random_grid(init: Grid):
    """Return a random grid from the argument by randomly fill all mutable cells."""
    N = len(init.data)
    return [
        [(random_generator.integers(1, N + 1) if cell == 0 else cell) for cell in row]
        for row in init.data
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
    main([None, "test.csv", "1024"])


def fitness(grid: Grid):
    """Determine grid's fitness score."""
    rows = grid.data
    cols = np.transpose(rows)
    subs = get_sub_grids(rows)
    # These three are negative
    rows_fitness = sum(fitness_sub(row) for row in rows)
    cols_fitness = sum(fitness_sub(col) for col in cols)
    subs_fitness = sum(fitness_sub(sub) for sub in subs)
    return rows_fitness + cols_fitness + subs_fitness


def fitness_sub(seq: list):
    """Return the number of unique elements in seq, excluding 0s since 0 cells are
    unfilled cells. The return value is positive."""
    uniq_elements = np.unique(seq)
    if 0 in seq:
        return len(uniq_elements) - 1
    else:
        return len(uniq_elements)


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
    0   0   0   1   1   1   2   2   2
    0   0   0   1   1   1   2   2   2
        ...
    3   3   3   4   4   4   5   5   5
        ...
    6   6   6   7   7   7   8   8   8"""
    return (row // sqrt_N) * sqrt_N + col // sqrt_N


# quick_test()

if __name__ == "__main__":
    main(sys.argv)
