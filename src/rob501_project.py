import numpy as np
import matplotlib.pyplot as plt
from imageio import imread

from .support.get_disparity import get_disparity
from .support.rectify_images import rectify_images
from .support.estimate_movement_scipy import estimate_movement

def run_project(start_frame, end_frame):
    It_1 = imread(f'./input/run1_base_hr/omni_image4/frame{start_frame}.png', as_gray = True)
    Ib_1 = imread(f'./input/run1_base_hr/omni_image5/frame{start_frame}.png', as_gray = True)
    
    It_2 = imread(f'./input/run1_base_hr/omni_image4/frame{end_frame}.png', as_gray = True)
    Ib_2 = imread(f'./input/run1_base_hr/omni_image5/frame{end_frame}.png', as_gray = True)
    
#     fig = plt.figure()
#     ax1 = fig.add_subplot(221)
#     ax2 = fig.add_subplot(222)  
#     ax3 = fig.add_subplot(223) 
#     ax4 = fig.add_subplot(224)  

#     ax1.imshow(It_1)
#     ax2.imshow(Ib_1)
#     ax3.imshow(It_2)
#     ax4.imshow(Ib_2)

#     plt.show()

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

    It_1, Ib_1 = rectify_images(It_1, Ib_1, Kt, Kb, dt, db, imageSize, T)
    disparity_1 = np.load('disparity.npy') #get_disparity(It_1, Ib_1, 70)
    
    It_2, Ib_2 = rectify_images(It_2, Ib_2,  Kt, Kb, dt, db, imageSize, T)
#     disparity_2 = get_disparity(It_2, Ib_2, 70)

    movement = estimate_movement(It_1, Ib_1, disparity_1, disparity_1, Kt, baseline)
    print(movement)


if __name__ == "__main__":
    run_project("000000_2018_09_04_17_19_42_773316", "000000_2018_09_04_17_19_42_773316") #"001577_2018_09_04_17_25_40_946193")