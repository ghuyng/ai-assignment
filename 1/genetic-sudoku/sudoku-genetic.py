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
            "\n"
            + reduce(
                operator.add,
                [
                    str(self.data[row][col])
                    + (" " if init[row][col] == 0 else ".")
                    + "\t"
                    for col in range(N)
                ],
            )
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
        print(np.average(fitness_weights))

        # Randomly select candidates by fitness weights
        selected = random.choices(
            population=population.candidates,
            weights=fitness_weights,
            k=population.population_size,
        )
        # print(list(map(fitness, selected)), "\n")
        # Breed selected candidates
        # children = one_point_crossover(selected)
        children = uniform_crossover(selected)
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


def uniform_crossover(candidates: list[Grid]) -> list[Grid]:
    N = len(candidates[0].data)
    children = []
    while candidates != []:
        child0 = candidates[0]
        child1 = candidates[1]
        for i in range(random_generator.integers(N)):
            point = random_generator.integers(N)
            child0.data[point], child1.data[point] = (
                child1.data[point],
                child0.data[point],
            )
        candidates = candidates[2:]
        children += [child0, child1]
    return children


def one_point_crossover(
    candidates: list[Grid], children: list[Grid] = []
) -> list[Grid]:
    """Perform "one_point_crossover" step for each pair in candidates.
    Return the accumulated children."""
    if candidates == []:
        return children
    else:
        N = len(candidates[0].data)
        # Identify the line to one_point_crossover
        point = random_generator.integers(1, N - 1)
        # Crossover: split at `point`, swap [0,point) and [point,N)
        # normal list concatenation does not work on numpy matrices
        child0 = Grid(
            np.concatenate((candidates[0].data[0:point], candidates[1].data[point:N]))
        )
        child1 = Grid(
            np.concatenate((candidates[1].data[0:point], candidates[0].data[point:N]))
        )
        return one_point_crossover(candidates[2:], children + [child0, child1])


def repeated_positions(seq: list):
    return [i for i in range(len(seq)) if seq[i] in seq[0:i]]


def mutate(grid: Grid, initial: Grid) -> Grid:
    """Perform "mutate" on grid, leave initial cells intact."""
    N = len(initial.data)
    # Randomly select a row and column
    for row in range(N):
        positions = repeated_positions(grid.data[row])
        if positions == []:
            cols = [random_generator.integers(N)]
        else:
            cols = positions
        for col in cols:
            # Replace with a random value unless the cell is immutable
            if initial.data[row][col] == 0:
                grid.data[row][col] = random_generator.integers(1, N + 1)
    return grid


def gen_random_grid(init: Grid) -> list[list[int]]:
    """Return a random grid from the argument by randomly fill all mutable cells."""
    N = len(init.data)
    return [
        [(random_generator.integers(1, N + 1) if cell == 0 else cell) for cell in row]
        for row in init.data
    ]


def read_initial(input_file) -> list[list[int]]:
    """Read initial numbers from input_file
    Return a list of cell lists; each cell, if initially declared, contains that number, else 0."""
    return [
        [(0 if cell == "" else int(cell)) for cell in row]
        for row in csv.reader(open(input_file))
    ]


def quick_test():
    """For REPL interaction."""
    main([None, "test-4x4.csv", "1024"])


def fitness(grid: Grid) -> int:
    """Determine grid's fitness score."""
    rows = grid.data
    cols = np.transpose(rows)
    subs = get_sub_grids(rows)
    rows_fitness = sum(map(fitness_sub, rows))
    cols_fitness = sum(map(fitness_sub, cols))
    subs_fitness = sum(map(fitness_sub, subs))
    return rows_fitness + cols_fitness + subs_fitness


def fitness_sub(seq: list) -> int:
    """Return the number of unique elements in seq, excluding 0s since 0 cells are
    unfilled cells. The return value is positive."""
    return len(set(seq) - {0})


def get_sub_grids(mat: list[list[int]]) -> list[list[int]]:
    """Return N sub squared grids from NxN grid."""
    N = len(mat)
    sqrt_N = int(np.sqrt(N))
    sub_grids_as_lists = [[] for _ in range(N)]
    for row in range(N):
        for col in range(N):
            sub_grids_as_lists[sub_grid_id(row, col, sqrt_N)].append(mat[row][col])
    return sub_grids_as_lists


def sub_grid_id(row, col, sqrt_N) -> int:
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
# pop4 = Population(read_initial("test-4x4.csv"), 4)

if __name__ == "__main__":
    main(sys.argv)
