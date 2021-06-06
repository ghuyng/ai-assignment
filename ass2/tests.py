#!/usr/bin/env python3

from CoGanh import *


def _gen_random_board() -> List[List[int]]:
    test_board_arr = (
        [-1 for _ in range(8)] + [1 for _ in range(8)] + [0 for _ in range(25 - 8 * 2)]
    )
    random.shuffle(test_board_arr)
    return [test_board_arr[5 * i : 5 * i + 5] for i in range(5)]


print(
    set(
        _get_surrounded(
            [
                [1, -1, 0, 0, 0],
                [1, -1, 0, 0, 0],
                [-1, 1, -1, 0, 0],
                [0, -1, 0, 0, 0],
                [0, 0, -1, 0, 0],
            ],
            1,
        )
    )
    == {(0, 0), (1, 0), (2, 1)}
)

print(
    set(
        _get_surrounded(
            [
                [1, -1, 0, 0, 0],
                [1, -1, 0, 0, 0],
                [-1, 0, 0, 0, 0],
                [1, -1, 0, 0, 0],
                [1, 1, -1, 0, 0],
            ],
            1,
        )
    )
    == {(0, 0), (1, 0), (3, 0), (4, 0), (4, 1)}
)


print(
    set(
        _get_surrounded(
            [
                [1, -1, 0, -1, 0],
                [1, -1, -1, 1, -1],
                [-1, 0, 0, -1, 0],
                [1, -1, 0, 0, 0],
                [1, 0, -1, 0, 0],
            ],
            1,
        )
    )
    == {(0, 0), (1, 0)}
)

print(
    set(
        _get_surrounded(
            [
                [-1, -1, -1, -1, -1],
                [-1, 1, 0, 1, -1],
                [-1, -1, -1, -1, -1],
                [-1, 0, 1, 1, -1],
                [-1, -1, -1, -1, -1],
            ],
            1,
        )
    )
    == set()
)

print(
    set(
        all_to_be_captured(
            (3, 1),
            (2, 2),
            [
                [-1, 1, -1, 0, 0],
                [0, 1, -1, -1, 1],
                [0, 1, 0, 1, 1],
                [0, -1, 0, 1, 1],
                [0, -1, -1, 0, 1],
            ],
            -1,
        )
    )
    == {(1, 1), (2, 1), (2, 3), (3, 3), (0, 1)}
)
