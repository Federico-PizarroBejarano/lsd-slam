import numpy as np
import cv2

def rectify_images(It, Ib):
    """
    Rectify two stereo images. Also applied a bilateral filter on both rectified images 
    to reduce noise while keeping edges sharp. 

    Parameters
    ----------
    It (np.ndarray): a numpy array representing the top image
    Ib (np.ndarray): a numpy array representing the bottom image
    
    Returns
    -------
    It_rect (np.ndarray): a numpy array representing the rectified top image
    Ib_rect (np.ndarray): a numpy array representing the rectified bottom image
    K_rect (np.ndarray): the new instrinic camera matrix
    """

    # Information needed for rectification
    Kt = np.array([[473.571, 0,  378.17],
          [0,  477.53, 212.577],
          [0,  0,  1]])
    Kb = np.array([[473.368, 0,  371.65],
          [0,  477.558, 204.79],
          [0,  0,  1]])
    dt = np.array([-0.333605,0.159377,6.11251e-05,4.90177e-05,-0.0460505])
    db = np.array([-0.3355,0.162877,4.34759e-05,2.72184e-05,-0.0472616])
    imageSize = (752, 480)
    baseline = 0.12
    T = np.array([0, baseline, 0])

    Rt, Rb, Pt, Pb = cv2.stereoRectify(Kt, dt, Kb, db, imageSize, np.eye(3), T, alpha=0)[0:4]

    map1_t, map2_t = cv2.initUndistortRectifyMap(Kt, dt, Rt, Pt, size=imageSize, m1type=0)
    map1_b, map2_b = cv2.initUndistortRectifyMap(Kb, db, Rb, Pb, size=imageSize, m1type=0)

    It_rect = cv2.remap(It, map1_t, map2_t, 0)
    Ib_rect = cv2.remap(Ib, map1_b, map2_b, 0)

    It_rect = cv2.bilateralFilter(It_rect,5,50,50)
    Ib_rect = cv2.bilateralFilter(Ib_rect,5,50,50)

    return It_rect, Ib_rect, Pt[0:3, 0:3]
