#!/usr/bin/env python3

from functools import reduce
import csv
import sys
import numpy as np
from dataclasses import dataclass
import random
import operator
from typing import List
import copy

random_generator = np.random.default_rng()

sqrt_squared = lambda arg: int(np.sqrt(arg))


@dataclass
class Grid:
    data: List[List[int]]

    def __init__(self, data: List[List[int]]):
        self.data = copy.deepcopy(data)
        self.N = len(data)

    def to_string(self, initial: List[List[int]]):
        rows_to_strs = [
            "\n"
            + reduce(
                operator.add,
                [
                    str(self.data[row][col])
                    + (" " if initial[row][col] == 0 else ".")
                    + "\t"
                    for col in range(self.N)
                ],
            )
            for row in range(self.N)
        ]
        return reduce(operator.add, rows_to_strs)


@dataclass
class Population:
    def __init__(self, initial: List[List[int]], population_size: int):
        self.initial = Grid(initial)
        self.N = len(initial)
        self.sqrtN = sqrt_squared(self.N)
        # There are N^2 cells, each can take -1 penalty from : row, column and sub grid constraints
        self.max_score = 2 * (self.N ** 2)
        self.population_size = population_size
        # Generate initial population
        self.candidates = [
            gen_random_grid(self.initial) for _ in range(self.population_size)
        ]


def transpose_raw_list(seq: List):
    return np.transpose(seq).tolist()


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
        print("Max:")
        print(
            population.candidates[
                fitness_weights.index(max(fitness_weights))
            ].to_string(population.initial.data)
        )
        print("Score:", max(fitness_weights))

        # Randomly select candidates by fitness weights
        # Breed selected candidates
        children = [
            child
            for pair in [
                crossover(
                    random.choices(
                        population=population.candidates, weights=fitness_weights, k=2
                    )
                )
                for _ in range(population.population_size // 2)
            ]
            for child in pair
        ]
        # Mutate the children and have them replace candidates
        population.candidates = sorted(
            population.candidates
            + [
                mutation
                for child in children
                for mutation in mutate_multiple(child, population.initial)
            ],
            key=fitness,
        )[-population.population_size :]
        # Re-calculate fitness weights
        fitness_weights = list(map(fitness, population.candidates))
        generation_no += 1

    # Get the perfect solution and display it
    solution = population.candidates[fitness_weights.index(population.max_score)]
    print(solution.to_string(population.initial.data))


def crossover(parents: List[Grid]) -> List[Grid]:
    """Accept a list of exactly TWO (2) grids."""
    n = sqrt_squared(parents[0].N)
    sum_scores_of_group = lambda data: [
        sum(map(fitness_sub, data[i * n : i * n + n])) for i in range(n)
    ]
    row_gr_scores = [sum_scores_of_group(p.data) for p in parents]
    parents_cols = [
        transpose_raw_list(parents[0].data),
        transpose_raw_list(parents[1].data),
    ]
    col_gr_scores = list(map(sum_scores_of_group, parents_cols))
    best_rows = [
        1 if row_gr_scores[1][i] > row_gr_scores[0][i] else 0 for i in range(n)
    ]
    best_cols = [
        1 if col_gr_scores[1][i] > col_gr_scores[0][i] else 0 for i in range(n)
    ]
    best_rows_data = []
    best_cols_data = []
    for i in range(n):
        lower_to_copy = i * n  # Inclusive
        upper_to_copy = i * n + n  # Exclusive
        best_rows_data += copy.deepcopy(
            parents[best_rows[i]].data[lower_to_copy:upper_to_copy]
        )
        best_cols_data += copy.deepcopy(
            parents_cols[best_cols[i]][lower_to_copy:upper_to_copy]
        )
    # c0, c1 = [Grid(best_rows_data), Grid(transpose_raw_list(best_cols_data))]
    # if False in list(map(test_subgrids, [c0, c1])):
    # print("ERROR by crossover!!!")
    return [Grid(best_rows_data), Grid(transpose_raw_list(best_cols_data))]


def test_subgrids(grid: Grid):
    N = grid.N
    n = sqrt_squared(N)
    for i in range(N):
        positions = get_subgrid_positions(N, i)
        values = [grid.data[row][col] for (row, col) in positions]
        if len(set(values)) != N:
            return False
    return True


def repeated_positions(seq: List):
    return [i for i in range(len(seq)) if seq[i] in seq[0:i]]


def mutate_multiple(grid: Grid, initial: Grid) -> List[Grid]:
    n = sqrt_squared(initial.N)
    offsprings = [mutate(Grid(grid.data), initial) for _ in range(initial.N)]
    offsprings.sort(key=fitness)
    return offsprings[-n:]


def mutate(grid: Grid, initial: Grid) -> Grid:
    """Perform "mutate" on grid, leave initial cells intact."""
    N = initial.N
    origin_grid = copy.deepcopy(grid.data)
    for subgrid_NO in range(N):
        fixed_positions, mutable_positions = fixed_and_mutable_positions_in_subgrid(
            initial, subgrid_NO
        )
        if len(mutable_positions) >= 2:
            # Select 2 random position that are mutable
            random.shuffle(mutable_positions)
            pos0, pos1 = mutable_positions[:2]
            # Swap them
            # My brain smoked when writing the following line, so don't ask me its meaning
            grid.data[pos0[0]][pos0[1]], grid.data[pos1[0]][pos1[1]] = (
                grid.data[pos1[0]][pos1[1]],
                grid.data[pos0[0]][pos0[1]],
            )
    # if not test_subgrids(grid):
    # print("ERROR by mutate!!!")
    return grid


def gen_random_grid(initial: Grid) -> Grid:
    """Return a random grid from the argument by filling all mutable cells.
    Sudoku's constraint for each subgrid is satisfied."""
    N = initial.N
    grid = Grid(copy.deepcopy(initial.data))
    for i in range(N):
        fill_subgrid(i, grid, initial)
    return grid


def fill_subgrid(subgrid_NO: int, grid: Grid, initial: Grid):
    """Fill grid's subgrid_NO-nth subgrid with unique for each cell.
    Respect initial."""
    N = initial.N
    fixed_positions, mutable_positions = fixed_and_mutable_positions_in_subgrid(
        grid, subgrid_NO
    )
    fixed_values = [initial.data[row][col] for (row, col) in fixed_positions]
    missing_values = list(set(range(1, N + 1)) - set(fixed_values))
    random.shuffle(missing_values)
    for (row, col) in mutable_positions:
        grid.data[row][col] = missing_values[0]
        missing_values = missing_values[1:]
    return grid


def fixed_and_mutable_positions_in_subgrid(initial_grid: Grid, subgrid_NO):
    N = initial_grid.N
    positions = get_subgrid_positions(N, subgrid_NO)
    fixed_positions = [
        (row, col) for (row, col) in positions if initial_grid.data[row][col] != 0
    ]
    mutable_positions = list(set(positions) - set(fixed_positions))
    return (fixed_positions, mutable_positions)


def get_subgrid_positions(N, subgrid_NO):
    n = sqrt_squared(N)
    rows = range(subgrid_NO // n * n, subgrid_NO // n * n + n)
    cols = range((subgrid_NO % n) * n, (subgrid_NO % n) * n + n)
    return [(row, col) for row in rows for col in cols]


def read_initial(input_file) -> List[List[int]]:
    """Read initial numbers from input_file
    Return a List of cell Lists; each cell, if initially declared, contains that number, else 0."""
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
    cols = transpose_raw_list(rows)
    rows_fitness = sum(map(fitness_sub, rows))
    cols_fitness = sum(map(fitness_sub, cols))
    return rows_fitness + cols_fitness


def fitness_sub(seq: List) -> int:
    """Return the number of unique elements in seq, excluding 0s since 0 cells are
    unfilled cells. The return value is positive."""
    return len(set(seq))


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
