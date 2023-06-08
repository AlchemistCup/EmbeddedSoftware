import numpy as np
import random
import cv2
from typing import Tuple

import generate_common as common

BOARD_DIMS = (8, 8)

def generate_board(frac_empty: float) -> Tuple[cv2.Mat, np.ndarray]:
    ref_board = np.full(BOARD_DIMS, -1)
    rows = []
    for i in range(BOARD_DIMS[0]):
        tiles = []
        for j in range(BOARD_DIMS[1]):
            is_empty = random.random() <= frac_empty
            if is_empty:
                tile = np.full(common.TILE_DIMS, 255, dtype=np.uint8)
            else:
                r, tile = common.generate_random_code()
                ref_board[i, j] = r
            tiles.append(common.add_black_border(tile))
        rows.append(np.hstack(tiles))
    
    board_img = np.vstack(rows)
    return board_img, ref_board

def main():
    for frac_tiles in [0.1, 0.25, 0.5, 0.75]:
        board, _ = generate_board(1-frac_tiles)
        cv2.imwrite(f'random_board_{int(frac_tiles * 100)}.png', common.add_black_border(board))

if __name__ == '__main__':
    main()