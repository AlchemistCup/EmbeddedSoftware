from picamera2 import Picamera2
import cv2
import numpy as np
from typing import List, Optional

from undistort import undistort

# Number of tiles of the board visible in an image
BOARD_WIDTH = 8
BOARD_HEIGHT = 8
RAW_IMG_DIMS = (1280, 720)

##### 1. Image capture
def setup_camera() -> Picamera2:
    picam = Picamera2()

    config = picam.create_preview_configuration() # Test if still config are better
    config['main']['size'] = RAW_IMG_DIMS
    config['main']['format'] = 'YUV420'
    picam.align_configuration(config)
    picam.configure(config)
    picam.start()

    return picam

def capture_img(picam: Picamera2) -> cv2.Mat:
    """
    Captures a grayscale image
    """
    img = picam.capture_array()
    return img[:RAW_IMG_DIMS[1], :RAW_IMG_DIMS[0]] # Requires image to be in YUV format

##### 2. Preprocessing
def preprocess_image(img: cv2.Mat) -> cv2.Mat:
    img_undistorted = undistort(img) # Remove fisheye distortion
    img_blur = cv2.GaussianBlur(img_undistorted, (5, 5), 1) # Remove gaussian noise
    img_threshold = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=11, C=2) # Potentially tweek params

    return img_threshold

##### 3. Board detection
# Returns an 8x8 array of coloured images of each board square
def find_board(img: cv2.Mat):
    img_preproc = preprocess_image(img)
    contours, hierarchy = cv2.findContours(img_preproc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find all top-level contours, encodes them as points of a polygon

    # Check if this is suitable with 4 cameras and potentially unclear board boundary
    def biggest_contour(contours):
        biggest = np.array([])
        max_area = 0
        for contour in contours: # Potentially slow loop
            area = cv2.contourArea(contour)
            if area > 50: # Small contour are noise
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area

        return biggest, max_area
    
    board_corners, _ = biggest_contour(contours)
    if board_corners.size == 0:
        print("[Warning] Couldn't find board contour")
        return None
    
    # Apply perspective transform (may not be necessary since camera position is fixed, always directly below board)
    # Returns a square image with the dimensions board_length_px of the square made by the board_corners on the image
    def isolate_board(img, board_corners, board_length_px):
        assert BOARD_WIDTH == BOARD_HEIGHT, "isolate_board assumes that our board segment is a square"
        # We need our corner points to be ordered in a fixed order to apply perspective transform
        # 2 different guides on google do it like this so it might be the best way???
        def reorder(board_corners):
            board_corners = board_corners.reshape((4, 2))
            sorted_corners = np.zeros((4, 1, 2), dtype=np.int32)
            add = board_corners.sum(1)
            sorted_corners[0] = board_corners[np.argmin(add)] # Top-left
            sorted_corners[3] = board_corners[np.argmax(add)] # Bottom-right
            diff = np.diff(board_corners, axis=1)
            sorted_corners[1] = board_corners[np.argmin(diff)] # Top-right
            sorted_corners[2] = board_corners[np.argmax(diff)] # Bottom-left

            return sorted_corners
        
        board_corners = reorder(board_corners)
        original_pts = np.float32(board_corners)
        target_pts = np.float32([0, 0], [board_length_px, 0], [0, board_length_px], [board_length_px, board_length_px])
        matrix = cv2.getPerspectiveTransform(original_pts, target_pts)

        board_img = cv2.warpPerspective(img, matrix, (board_length_px, board_length_px))
        # Need to do some post processing here to binarise image to facilitate QR code detection later. Better to do this in one go now before chopping it up into squares
        print("[Warning] Not doing any post processing to board_img")
        # Current idea:
        # grey_board = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # _, binary_board = cv2.threshold(grey_board, 150, 255, cv2.THRESH_BINARY) # Simple thresholding probably sufficient here
        return board_img
    
    # Not sure how big the board will be, need to measure size when setup is ready
    # Note that this cropped version is a coloured image of the board!
    board = isolate_board(img, board_corners, board_length_px = 704)

##### 4. Board state detection
def detect_board_state(board_img):
    # Very simple approach, just splits up the image evenly into an 8x8 grid (all squares have the same size so this should be fine) -- REQUIRES BOARD TO ONLY INCLUDE 8x8 squares!!!!
    def segment(board_img):
        assert board_img.shape[0] == board_img.shape[1], f"Board image is not a square, dimensions {board_img.shape}"
        assert len(board_img.shape) == 2, f"Board image is not 2D (contains multiple channels), dimensions {board_img.shape}"

        n = board_img.shape[0]
        assert n % 8 == 0, f"Board image dimensions {board_img.shape} is not divisible into 8x8 grid" 
        subarray_size = n // 8
        squares = board_img.reshape((8, subarray_size, 8, subarray_size)).transpose((0, 2, 1, 3)).reshape((8, 8, subarray_size, subarray_size))

        # Old approach, should be slower since using python loop
        # squares = []
        # rows = np.vsplit(board_img, 8)
        # for row in rows:
        #     squares.append(np.hsplit(row, 8))
        # squares = np.array(squares)
        
        # Check dimensionality of this works out
        return squares
    
    squares = segment(board_img)
    # Potentially refactor this to be numpy array using ints?
    # -1 can indicate emptiness
    # board = np.empty((8, 8), dtype=np.int8)
    # board = np.array([decode_qr_code(grid[idx]) for idx in np.ndindex(grid.shape[:2])], dtype=np.int8).reshape((8, 8))

    board: List[List[Optional[str]]] = [[None] * 8 for _ in range(8)]
    qcd = cv2.QRCodeDetector()
    for idx, square in np.ndenumerate(squares):
        res, _, _ = qcd.detectAndDecode(square)
        if res: # Tile on square
            if res % 27 == 26: # Blank tile
                char = '?'
            else:
                char = chr(ord('A') + (res % 27))
            
            board[idx[0]][idx[1]] = char

    return board

def main():
    picam = setup_camera()

    while True:
        img: cv2.Mat = capture_img(picam)
        board_img: cv2.Mat = find_board(img)
        board_state: List[List[Optional[str]]] = detect_board_state(board_img)
        print(board_state)

if __name__ == "__main__":
    main()   