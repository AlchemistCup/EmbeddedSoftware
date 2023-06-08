import numpy as np
from time import time
from typing import Dict, Callable
import random
import cv2

from generate_rack import generate_rack

def calculate_f1(result: Dict[str, int], reference: Dict[str, int]):
    tp = 0
    fp = 0
    fn = 0

    for key in reference:
        if key in result:
            tp += min(result[key], reference[key])
            fp += max(result[key] - reference[key], 0)
            fn += max(reference[key] - result[key], 0)
        else:
            fn += reference[key]

    for key in result:
        if key not in reference:
            fp += result[key]

    precision = tp / (tp + fp) if tp + fp > 0 else 1.
    recall = tp / (tp + fn) if tp + fn > 0 else 1.
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return f1

def benchmark(iterations: int, n_of_tiles: int, detect_rack: Callable[[cv2.Mat], Dict[str, int]]):
    times = np.empty((iterations,), dtype=np.float32)
    f1_scores = np.empty((iterations,), dtype=np.float32)
    for i in range(iterations):
        rack, ref = generate_rack(n_of_tiles)

        start = time()
        res = detect_rack(rack)
        end = time()

        times[i] = (end - start) * 1000
        f1_scores[i] = calculate_f1(res, ref)
    
    print(f"Benchmark complete ({iterations} iterations): ")
    print(f"\tTime: mean = {times.mean():.5f}ms, std. dev = {times.std():.5f}ms")
    print(f"\tF1 Score: mean = {f1_scores.mean():.5f}, std. dev = {f1_scores.std():.5f}")

def full_benchmark(iterations: int, detect_rack: Callable[[cv2.Mat], Dict[str, int]]):
    for n_of_tiles in range(8):
        print(f"Testing {n_of_tiles} tiles")
        benchmark(iterations, n_of_tiles, detect_rack)


