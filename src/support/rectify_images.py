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


if __name__ == "__main__":
    # Load the stereo images
    It = imread('./input/run1_base_hr/omni_image4/frame000000_2018_09_04_17_19_42_773316.png', as_gray = True)
    Ib = imread('./input/run1_base_hr/omni_image5/frame000000_2018_09_04_17_19_42_773316.png', as_gray = True)

    Kt = np.array([[473.571, 0,  378.17],
          [0,  477.53, 212.577],
          [0,  0,  1]])
    Kb = np.array([[473.368, 0,  371.65],
          [0,  477.558, 204.79],
          [0,  0,  1]])
    dt = np.array([-0.333605,0.159377,6.11251e-05,4.90177e-05,-0.0460505])
    db = np.array([-0.3355,0.162877,4.34759e-05,2.72184e-05,-0.0472616])

    imageSize = (752, 480)
    T = np.array([0, 0.12, 0])

    It_rect, Ib_rect, K = rectify_images(It, Ib, Kt, Kb, dt, db, imageSize, T)

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)  
    ax3 = fig.add_subplot(223) 
    ax4 = fig.add_subplot(224)  

    ax1.imshow(It, cmap='gray')
    ax1.title.set_text("Top Image")
    ax2.imshow(Ib, cmap='gray')
    ax2.title.set_text("Bottom Image")
    ax3.imshow(It_rect, cmap='gray')
    ax3.title.set_text("Rectified Top Image")
    ax4.imshow(Ib_rect, cmap='gray')
    ax4.title.set_text("Rectified Bottom Image")

    plt.show()