import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
import cv2

def rectify_images(It, Ib, Kt, Kb, dt, db, imageSize, T):
    """
    Rectify two stereo images given their camera matrices, distortion matrices, desired output
    image size, and translation between the two stereo cameras. Also applied a bilateral filter
    on both rectified images to reduce noise while keeping edges sharp. 

    Parameters
    ----------
    It (np.ndarray): a numpy array representing the top image
    Ib (np.ndarray): a numpy array representing the bottom image
    Kt (np.ndarray): the top camera instrinic camera matrix
    Kb (np.ndarray): the bottom camera instrinic camera matrix
    dt (np.ndarray): the distortion parameters array for the top camera
    db (np.ndarray): the distortion parameters array for the bottom camera
    imageSize (tuple): a tuple representing the inverted desired output size of the rectified images
    T (np.ndarray): a translation vector between cameras
    
    Returns
    -------
    It_rect (np.ndarray): a numpy array representing the rectified top image
    Ib_rect (np.ndarray): a numpy array representing the rectified bottom image
    K_rect (np.ndarray): the new instrinic camera matrix
    """

    Rt, Rb, Pt, Pb = cv2.stereoRectify(Kt, dt, Kb, db, imageSize, np.eye(3), T, alpha=0)[0:4]

    map1_t, map2_t = cv2.initUndistortRectifyMap(Kt, dt, Rt, Pt, size=imageSize, m1type=0)
    map1_b, map2_b = cv2.initUndistortRectifyMap(Kb, db, Rb, Pb, size=imageSize, m1type=0)

    It_rect = cv2.remap(It, map1_t, map2_t, 0)
    Ib_rect = cv2.remap(Ib, map1_b, map2_b, 0)

    It_rect = cv2.bilateralFilter(It_rect,5,50,50)
    Ib_rect = cv2.bilateralFilter(Ib_rect,5,50,50)

    return It_rect, Ib_rect, Pt[0:3, 0:3]
