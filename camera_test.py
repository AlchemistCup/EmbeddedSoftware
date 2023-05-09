from picamera2 import Picamera2, Preview
import time
import cv2

from undistort import undistort

def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration()
    config['main']['size'] = (1280,720)
    config['main']['format'] = "RGB888" # There does not appear to be a way to capture greyscale images directly, even example docs use cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    picam2.align_configuration(config)
    print(config)
    picam2.configure(config)
    picam2.start()

    while True:
        start = time.time()
        im = picam2.capture_array() # Can grab frames at around 30FPS
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = undistort(im) # Optimised undistort, now achieve 10-25FPS (~25ms)
        # Raspberry Pi 4B is ~3x more powerful than 3B+ (current model) + can have more RAM (would recommend 4GB for image processing) => much faster FPS
        end = time.time()
        frame_capture_time = end - start
        fps = 1 / frame_capture_time
        print(int(fps))

if __name__ == '__main__':
    main()