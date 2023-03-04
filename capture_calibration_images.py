import cv2
from picamera2 import Picamera2
from pathlib import Path

CALIBRATION_DIR = Path(__file__).parent.resolve() / 'fisheye_calibration'

def main():
    # Setup camera
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (1280,720)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()

    i = 1
    while True:
        im = picam2.capture_array()
        cv2.imshow("Camera", im)

        keypress = cv2.waitKey(1)
        if keypress == ord('q'):
            break
        elif keypress == ord('c'):
            path = CALIBRATION_DIR / f'img_{i:03}.jpg'
            cv2.imwrite(path, im)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
