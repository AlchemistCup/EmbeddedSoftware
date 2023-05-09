import cv2
import numpy as np
import sys
from pathlib import Path
import time

# Calibrated to current lens setup
DIM=(1280, 720)
K=np.array([[375.42727147451, 0.0, 641.496818682067], [0.0, 375.3825372354678, 372.2592486673318], [0.0, 0.0, 1.0]])
D=np.array([[-0.009176217948661868], [-0.00876731233656425], [0.006551298173650552], [-0.0019715619325778844]])

def main():
    for p in sys.argv[1:]:
        undistort_and_display(Path(p))

def undistort(img, balance=0.3):
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort    
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"    
    
    # Hardcode all dims to be identical for now
    dim2 = dim1   
    dim3 = dim1 
    
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0    
    
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return undistorted_img    

def undistort_and_display(img_path, balance=0.3, dim2=None, dim3=None):    
    print(img_path)
    img = cv2.imread(str(img_path))
    
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