import time
import cv2
from pathlib import Path

from utils import find_starting_id
from board_capture import setup_camera, preprocess_image, setup_camera_gray

TEST_DIR = Path(__file__).parent.resolve() / 'test_img'

def main():
    picam = setup_camera()
    i = find_starting_id(TEST_DIR)

    n = 0
    total = 0
    while True:
        start = time.time()
        img = picam.capture_array() # Can grab frames at around 30FPS
        img_preproc = preprocess_image(img)
        end = time.time()

        frame_capture_time = end - start
        fps = int(1 / frame_capture_time)
        total += fps
        n += 1
        if n % 10 == 0:
            print(f"[Info] Averaging {(total / n):.2f} fps over {n} frames")
        # pos = (30, 60)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img_preproc, str(fps), pos, font, fontScale=1, color=255, thickness=1)

        img_final = img_preproc
        cv2.imshow("Camera", img_final)
        keypress = cv2.waitKey(1)
        if keypress == ord('q'):
            print(f"[Info] Averaged {(total / n):.2f} fps over {n} frames")
            break
        elif keypress == ord('s'):
            path = TEST_DIR / f'img_{i:03}.jpg'
            original_path = TEST_DIR / f'raw_img_{i:03}.jpg'
            print(f'[Info] Saving image to {path}')
            cv2.imwrite(str(path), img_final)
            cv2.imwrite(str(original_path), img)
            i += 1

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()