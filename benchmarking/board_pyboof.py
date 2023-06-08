import cv2
import numpy as np
import pyboof as pb
from typing import List, Optional
from board_benchmark import full_benchmark

qcd = pb.FactoryFiducial(np.uint8).qrcode()

def main():
    iterations = 100
    print("Testing iterative method")
    full_benchmark(iterations, detect_iterative)

    print("Testing simultaneous method")
    full_benchmark(iterations, detect_simultaneous)

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
            # Requires a copy of square due to non-contiguous memory (square is a view on original array)
            img = pb.ndarray_to_boof(square.copy()) 
            qcd.detect(img)
            if len(qcd.detections) == 1: # Tile on square
                board[i, j] = qcd.detections[0].message

    return board

def detect_vectorised(board_img):
    def segment(board_img):
        n = board_img.shape[0]
        subarray_size = n // 8
        squares = board_img.reshape((8, subarray_size, 8, subarray_size)).transpose((0, 2, 1, 3)).reshape((64, subarray_size, subarray_size))
        
        return squares
    
    def decode_qr_vec(square):
        img = pb.ndarray_to_boof(square)
        qcd.detect(img)
        if len(qcd.detections) == 1:
            return np.int16(qcd.detections[0].message)
        else:
            return np.int16(-1)
    
    squares = segment(board_img)
    board = np.vectorize(decode_qr_vec, signature='(n,n)->()')(squares).reshape(8,8)

    return board

def detect_simultaneous(board_img):
    n = board_img.shape[0]
    subarray_size = n // 8
    def get_index(bounding_box):
        # assert len(bounding_box) == 4 we need 4 points
        bounding_box = np.array(bounding_box.convert_tuple())
        y, x = bounding_box.mean(axis=0) // subarray_size
        return int(x), int(y)

    board = np.full((8, 8), -1, dtype=np.int16)
    board_img = pb.ndarray_to_boof(board_img)
    qcd.detect(board_img)
    for detection in qcd.detections:
        idx = get_index(detection.bounds)
        if detection.message:
            board[idx] = detection.message
    
    return board


if __name__ == '__main__':
    main()