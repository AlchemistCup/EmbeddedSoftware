import cv2
from picamera2 import Picamera2
from pathlib import Path
from utils import find_starting_id

CALIBRATION_DIR = Path(__file__).parent.resolve() / 'fisheye_calibration'

def main():
    # Setup camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration()
    config['main']['size'] = (1280,720)
    config['main']['format'] = "RGB888" # There does not appear to be a way to capture greyscale images directly, even example docs use cv2.cvtColor(im, cv2.COLOR_BRG2GRAY)
    picam2.align_configuration(config)
    print(config)
    picam2.configure(config)
    picam2.start()

    i = find_starting_id(CALIBRATION_DIR)
    while True:
        im = picam2.capture_array()
        cv2.imshow("Camera", im)

        keypress = cv2.waitKey(1)
        if keypress == ord('q'):
            break
        elif keypress == ord('c'):
            path = CALIBRATION_DIR / f'img_{i:03}.jpg'
            print(f'[Info] Saving image to {path}')
            cv2.imwrite(str(path), im)
            i += 1
        elif keypress != -1:
            print(keypress)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
