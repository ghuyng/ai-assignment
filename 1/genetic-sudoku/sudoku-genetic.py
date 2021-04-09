#!/usr/bin/env python3

import csv
import sys


def main(args):
    for row in read_initial(args[1]):
        print(row)


def read_initial(input_file):
    return [row for row in csv.reader(open(input_file))]


if __name__ == "__main__":
    main(sys.argv)
