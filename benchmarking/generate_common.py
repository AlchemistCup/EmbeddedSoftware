import cv2
import numpy as np
import segno
from time import time
from random import randint, random
from typing import Tuple, Callable

TEST_FILE = 'qr_test.png'
TILE_DIMS = (88, 88)
CODE_RANGE = (0, 999)

def generate_random_code(range: Tuple[int, int] = CODE_RANGE):
    r = randint(*range)
    qrcode = segno.make(r, version=1)
    assert qrcode.error == 'H'

    # Load QR code as cv2 image
    qrcode.save(TEST_FILE, scale = 5)
    img = cv2.imread(TEST_FILE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, TILE_DIMS)

    return r, img

# Currently just pixelates image
def generate_random_noisy_code(range: Tuple[int, int] = CODE_RANGE):
    r = randint(*range)
    qrcode = segno.make(r, version=1)
    assert qrcode.error == 'H'

    arr = np.array(qrcode.matrix, dtype=np.uint8)
    arr[arr == 0] = 255
    arr[arr == 1] = 0
    img = cv2.copyMakeBorder(arr, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255)
    img = cv2.resize(img, TILE_DIMS)

    return r, img

def add_black_border(img):
    bordered_img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    return bordered_img