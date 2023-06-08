import numpy as np
from time import time
from typing import Callable

from generate_board import BOARD_DIMS, generate_board

def find_differences(result, reference):
    diff_idx = np.where(result != reference)
    row_idx, col_idx = diff_idx

    differences = [
        f"Obtained {res} instead of {ref} @ ({row}, {col})"
        for res, ref, row, col in zip(result[diff_idx], reference[diff_idx], row_idx, col_idx)
    ]

    return differences

def calculate_accuracy(errors):
    total = BOARD_DIMS[1] * BOARD_DIMS[0]
    return (total - errors) / total * 100

def benchmark(iterations: int, frac_empty: float, detect_board: Callable):
    times = []
    n_of_errors = []
    for _ in range(iterations):
        grid, ref = generate_board(frac_empty)

        start = time()
        res = detect_board(grid)
        end = time()

        times.append((end - start) * 1000)
        if not (res == ref).all():
            errors = find_differences(res, ref)
            n_of_errors.append(len(errors))
        else:
           n_of_errors.append(0)

    times = np.array(times)
    n_of_errors = np.array(n_of_errors)
    accuracy = np.vectorize(calculate_accuracy)(n_of_errors)

    print(f"Benchmark complete ({iterations} iterations): ")
    print(f"\tTime: mean = {times.mean():.5f}ms, std. dev = {times.std():.5f}ms")
    print(f"\tAccuracy: mean = {accuracy.mean():.5f}%, std. dev = {accuracy.std():.5f}%")

def full_benchmark(iterations: int, detect_grid: Callable):
    for frac_tiles in [0., 0.1, 0.25, 0.5, 0.75, 0.9]:
        print(f"Testing {frac_tiles * 100}% tiles")
        benchmark(iterations, 1 - frac_tiles, detect_grid)
