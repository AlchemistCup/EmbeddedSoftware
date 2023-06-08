from picamera2 import Picamera2
import numpy as np
import cv2
from board_capture import setup_camera
import asyncio

def on_complete(job):
    print(f"Took picture, callback called with: {job}")
    print(f"Result: {job.get_result()}")

async def main():
    picam = setup_camera()
    img: cv2.Mat = picam.capture_array(signal_function=on_complete)
    print("Not blocking execution")
    while True:
        await asyncio.sleep(0.01)

if __name__ == '__main__':
    asyncio.run(main())
