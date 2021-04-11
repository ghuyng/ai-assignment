#!/usr/bin/env python3

import csv
import sys
import numpy as np
from dataclasses import dataclass
import random

from typing import List
# TODO: make it configurable
POPULATION_SIZE = 1000

random_generator = np.random.default_rng()


@dataclass
class Grid:
    def __init__(self, grid: List[List[int]], initial=None):
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
    solve1(grid)

def solve1(grid):
    """A sudoku solver, used for stratery pattern."""
    MUTATION_PROB = 20
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

        print (" Max score = ", max(fitness_weights))
        print (" Mean = ", sum(fitness_weights) / len(fitness_weights))
        next_gen = sorted(candidates, key=fitness)[POPULATION_SIZE*90//100 :]
        # next_gen = random.choices(
        #     population=candidates, weights=fitness_weights, k=POPULATION_SIZE * 10 // 100)

        for _ in range(POPULATION_SIZE * 45 // 100):
            # Randomly select candidates by fitness weights
            selected = random.choices(
                population=candidates, weights=fitness_weights, k=2
            )
            # Breed selected candidates
            # child1, child2 = crossover(selected)
            child1, child2 = uniform_crossover(selected)
            # Mutate the children and have them replace candidates
            # candidates = list(map(mutate, children))
            if random_generator.integers(100) < MUTATION_PROB:
                child1 = mutate(child1)
            if random_generator.integers(100) < MUTATION_PROB:
                child2 = mutate(child2)

            next_gen+= [child1, child2]

        candidates = next_gen
        # Re-calculate fitness weights
        fitness_weights = list(map(fitness, candidates))
        if (generation_no > 2000):
            raise Exception("No solution")
        # if (generation_no == 100):
        #     MUTATION_PROB = 10
        elif (generation_no == 200):
            MUTATION_PROB = 10
        generation_no += 1

    # Get the perfect solution and display it
    solution = candidates[fitness_weights.index(grid.max_score)]
    print_grid(solution)


def crossover(grids: List[Grid], children: List[Grid] = []) -> List[Grid]:
    """Perform "crossover" step for each pair in grids.
    Return the accumulated children."""
    if grids == []:
        return children
    else:
        N = grids[0].N
        c1, c2 = [], []
        for index, row in enumerate(grids[0].current):
            point = random_generator.integers(N - 1)
            c1 += list(row[:point]) + list(grids[1].current[index][point:])
            c2 += list(grids[1].current[index][:point]) + list(row[point:])

        child1 = Grid(np.array(c1).reshape(N, N), grids[0].initial)
        child2 = Grid(np.array(c2).reshape(N, N), grids[0].initial)
        return child1, child2

def uniform_crossover(parents):
    """Uniform crossover: Swap genes at random points instead of exchanging segments."""
    c1, c2 = [], []
    N = parents[0].N
    for i in range(N):
        for j in range(N):
            point = random_generator.integers(100)
            if (point < 50):
                c1 += [parents[0].current[i][j]]
                c2 += [parents[1].current[i][j]]
            else:
                c1 += [parents[1].current[i][j]]
                c2 += [parents[0].current[i][j]]

    child1 = Grid(np.array(c1).reshape(N, N), parents[0].initial)
    child2 = Grid(np.array(c2).reshape(N, N), parents[0].initial)
    return child1, child2

def mutate(grid: Grid) -> Grid:
    """Perform "mutate" on grid."""
    # Randomly select a row and column
    # row = random_generator.integers(grid.N)
    for row in range(grid.N):
        col = random_generator.integers(grid.N)
        # Replace with a random value unless the cell is immutable
        if grid.initial[row][col] == 0:
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


def gen_random_grid(grid: List[List[int]]):
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

def fitness(grid: Grid):
    """Determine grid's fitness score."""
    rows = grid.current
    cols = np.transpose(rows)
    subs = get_sub_grids(rows)
    # These three are negative
    rows_fitness = sum(fitness_sub(row) for row in rows)
    cols_fitness = sum(fitness_sub(col) for col in cols)
    subs_fitness = sum(fitness_sub(sub) for sub in subs)
    return rows_fitness + cols_fitness + subs_fitness


def fitness_sub(seq: List):
    """Return the number of unique elements."""
    uniq_elements = np.unique(seq)
    return len(uniq_elements)


def get_sub_grids(mat: List[List[int]]) -> List[List[int]]:
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

if __name__ == "__main__":
    main(sys.argv)
