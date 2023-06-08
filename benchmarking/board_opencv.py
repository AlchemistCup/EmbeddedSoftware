import cv2
import numpy as np
from typing import List, Optional
from board_benchmark import full_benchmark

qcd = cv2.QRCodeDetector()

def main():
    iterations = 1000
    # print("Testing simultaneous method")
    # full_benchmark(iterations, detect_simultaneous)

    print("Testing vectorised method")
    full_benchmark(iterations, detect_vectorise)

    print("Testing iterative method")
    full_benchmark(iterations, detect_iterative)

def detect_iterative(board_img):
    def segment(board_img):
        squares = []
        rows = np.vsplit(board_img, 8)
        for row in rows:
            squares.append(np.hsplit(row, 8))
        
        return squares
    
    squares = segment(board_img)
    board = np.full((8, 8), -1, dtype=np.int16)

    for i, row in enumerate(squares):
        for j, square in enumerate(row):
            res, _, _ = qcd.detectAndDecode(square)
            if res: # Tile on square
                board[i, j] = res

    return board

def detect_vectorise(board_img):
    def segment(board_img):
        n = board_img.shape[0]
        subarray_size = n // 8
        squares = board_img.reshape((8, subarray_size, 8, subarray_size)).transpose((0, 2, 1, 3)).reshape((64, subarray_size, subarray_size))
        
        return squares
    
    def decode_qr_vec(square):
        res, _, _ = qcd.detectAndDecode(square)
        return np.int16(res) if res else np.int16(-1)
    
    squares = segment(board_img)
    board = np.vectorize(decode_qr_vec, signature='(n,n)->()')(squares).reshape(8,8)

    return board

def detect_simultaneous(board_img):
    n = board_img.shape[0]
    subarray_size = n // 8
    def get_index(bounding_box):
        # assert len(bounding_box) == 4 we need 4 points
        col, row = bounding_box.mean(axis=0) // subarray_size
        return int(row), int(col)

    board = np.full((8, 8), -1, dtype=np.int16)
    res, values, points, _ = qcd.detectAndDecodeMulti(board_img)
    if res:
        for value, bounding_box in zip(values, points):
            idx = get_index(bounding_box)
            if value:
                board[idx] = value
    
    return board


if __name__ == '__main__':
    main()