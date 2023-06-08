import numpy as np
import random
import cv2
from typing import Tuple, Dict
from time import time

import generate_common as common
from generate_common import TILE_DIMS

RACK_WIDTH = 860
assert RACK_WIDTH >= TILE_DIMS[1] * 7

def generate_rack(n_of_tiles: int) -> Tuple[cv2.Mat, Dict[str, int]]:
    tiles = []
    rack_values = {}
    for i in range(n_of_tiles):
        r, tile = common.generate_random_code()
        r = str(r)
        tiles.append(tile)
        rack_values.setdefault(r, 0)
        rack_values[r] += 1

    rack_img = randomly_place_tiles(tiles)
    return rack_img, rack_values

class Gap:
    def __init__(self, start, end) -> None:
        self._start = start
        self._end = end
        self._capacity = max(int((end - start) // TILE_DIMS[1]), 0)

    @property
    def start(self):
        return self._start
    
    @property
    def end(self):
        return self._end
    
    @property
    def capacity(self):
        return self._capacity
    
    @property
    def range(self):
        return self.end - self.start
    
    def get_random_start_point(self, required_remaining_capacity):
        assert required_remaining_capacity <= self.capacity - 1

        if self.capacity - 1 > required_remaining_capacity:
            return random.randint(self.start, self.end - TILE_DIMS[1])
        
        starts = np.arange(self.start, self.end - TILE_DIMS[1] + 1)


        def is_valid_start(point):
            return ((point - self.start) // TILE_DIMS[1]) + ((self.end - (point + TILE_DIMS[1])) // TILE_DIMS[1]) >= required_remaining_capacity
        
        assert is_valid_start(starts).any()
        return random.choice(starts[is_valid_start(starts)])
    
    def split(self, start_point):
        left = Gap(self.start, start_point)
        right = Gap(start_point + TILE_DIMS[1], self.end)
        new_gaps = filter(lambda g: g.capacity > 0, [left, right])
        return list(new_gaps)
    
    def __repr__(self) -> str:
        return f"Gap([{self.start}:{self.end}], capacity = {self.capacity})"

def randomly_place_tiles(tiles):
    rack = np.ones((TILE_DIMS[0], RACK_WIDTH), dtype=np.uint8) * 255
    random.shuffle(tiles)

    gaps = [Gap(0, RACK_WIDTH)]
    for i, tile in enumerate(tiles):
        # Generate random valid insertion point
        remaining_tiles = len(tiles) - i - 1
        gap_idx = np.random.choice(np.arange(0, len(gaps)), p=np.array([g.range for g in gaps]) / sum(g.range for g in gaps))
        gap = gaps[gap_idx]
        remaining_capacity = sum(g.capacity for g in gaps) - gap.capacity
        required_capacity_post_insertion = max(remaining_tiles - remaining_capacity, 0)
        start = gap.get_random_start_point(required_capacity_post_insertion)

        # Update gaps state
        new_gaps = gap.split(start)
        if len(new_gaps) > 0:
            gaps[gap_idx] = new_gaps[0]
            for gap in new_gaps[1:]:
                gaps.append(gap)
        else:
            del gaps[gap_idx]

        # Insert tile into container
        rack[:, start:start+TILE_DIMS[1]] = tile

    return rack

def main():
    img, ref = generate_rack(7)
    cv2.imwrite('test_rack.png', img)
    print(ref)

    for n in range(8):
        times = np.empty((100,), dtype=np.float32)
        for i in range(100):
            start = time()
            generate_rack(n)
            end = time()
            times[i] = (end - start) * 1000
        print(f'Took an average of {times.mean():.5f}ms to run for {n} tiles')

if __name__ == '__main__':
    main()
