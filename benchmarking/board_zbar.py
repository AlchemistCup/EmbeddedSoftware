import numpy as np
from pyzbar import pyzbar
from board_benchmark import full_benchmark

def main():
    iterations = 100
    print("Testing simultaneous method")
    full_benchmark(iterations, detect_simultaneous)

    #print("Testing vectorised method")
    #full_benchmark(iterations, detect_vectorise)

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
            barcodes = pyzbar.decode(square)
            if len(barcodes) == 1: # Tile on square
                board[i, j] = barcodes[0].data

    return board

def detect_vectorise(board_img):
    def segment(board_img):
        n = board_img.shape[0]
        subarray_size = n // 8
        squares = board_img.reshape((8, subarray_size, 8, subarray_size)).transpose((0, 2, 1, 3)).reshape((64, subarray_size, subarray_size))
        
        return squares
    
    def decode_qr_vec(square):
        barcodes = pyzbar.decode(square)
        if len(barcodes) == 1:
            return np.int16(barcodes[0].data)
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
        y, x = bounding_box.mean(axis=0) // subarray_size
        return int(x), int(y)

    board = np.full((8, 8), -1, dtype=np.int16)
    barcodes = pyzbar.decode(board_img)
    for code in barcodes:
        idx = get_index(np.array(code.polygon))
        board[idx] = code.data
    
    return board

if __name__ == '__main__':
    main()