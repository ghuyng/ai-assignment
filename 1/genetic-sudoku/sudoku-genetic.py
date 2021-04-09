#!/usr/bin/env python3

import csv
import sys


def main(args):
    grid = read_initial(args[1])
    N = len(grid)


def read_initial(input_file):
    """Read initial numbers from input_file
    Return a list of cell lists; each cell, if initially declared, is that number, else 0"""
    return [
        [(0 if cell == "" else int(cell)) for cell in row]
        for row in csv.reader(open(input_file))
    ]


def quick_test():
    main([None, "test.csv"])


if __name__ == "__main__":
    main(sys.argv)
