import cv2
import numpy as np
import sys
from pathlib import Path
import time

# TODO: Refactor this mess
# class FishEyeCorrector:
#     def __init__(self, dim: Tuple[int, int], k: np.array, d: np.array):
# 
# Maybe use @cached_property to memoise constant params we need to calculate (https://stackoverflow.com/a/56097764)

# Logic adapted from https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-part-2-13990f1b157f

# Calibrated to current lens setup
DIM=(1280, 720)
K=np.array([[375.42727147451, 0.0, 641.496818682067], [0.0, 375.3825372354678, 372.2592486673318], [0.0, 0.0, 1.0]])
D=np.array([[-0.009176217948661868], [-0.00876731233656425], [0.006551298173650552], [-0.0019715619325778844]])

# Compute rectification map on startup to speedup undistortion in production
# This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image.
balance = 0.3
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, DIM, np.eye(3), balance=balance)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, DIM, cv2.CV_16SC2) # Bulk of time spent here

def main():
    for p in sys.argv[1:]:
        undistort_and_display(Path(p))

def undistort(img):
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Crop image
    w_crop = 200
    undistorted_img = undistorted_img[:, w_crop:DIM[0]-w_crop]    

    return undistorted_img    

def undistort_and_display(img_path, balance=0.3, dim2=DIM, dim3=DIM):
    """
    Used to test different parameters for undistortion. Recomputes rectification map when called - DO NOT USE IN PRODUCTION.
    """    
    print(img_path)
    img = cv2.imread(str(img_path))
    if img.shape[:2] != DIM:
        print(f'[WARNING] Image dimensions {img.shape[:2]} do not match calibration dimensions {DIM}')
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2) # Bulk of time spent here
    
    start = time.time()
    undistorted_img = undistort(img)
    end = time.time()
    print(f"Took {end-start} to undistort")
    
    cv2.imshow("undistorted", undistorted_img)
    keypress = cv2.waitKey(0)
    if keypress == ord('s'):
        path = img_path.parent / f"{img_path.stem}_undistorted{img_path.suffix}"
        cv2.imwrite(str(path), undistorted_img)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':    
    main()