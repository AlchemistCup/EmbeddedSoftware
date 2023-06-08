import cv2
import numpy as np
from typing import List, Optional
from rack_benchmark import full_benchmark

qcd = cv2.QRCodeDetector()

def main():
    iterations = 1000
    full_benchmark(iterations, detect)

def detect(rack_img: cv2.Mat):
    rack = {}
    res, values, _, _ = qcd.detectAndDecodeMulti(rack_img)
    if res:
        for value in values:
            if value:
                rack.setdefault(value, 0)
                rack[value] += 1

    return rack


if __name__ == '__main__':
    main()