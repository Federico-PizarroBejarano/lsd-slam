import os
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from scipy.spatial.transform import Rotation as R

from .support.get_disparity import get_disparity
from .support.rectify_images import rectify_images
from .support.estimate_movement import estimate_movement
from .support.plot_path import plot_path
from .support.get_utm_poses import get_utm_poses


def run_project(start_frame, end_frame):
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
    maxd = 70

    num_images = int(end_frame) - int(start_frame)
    all_movements = np.zeros((num_images+1, 6))

    ground_truth = np.genfromtxt("./input/run1_base_hr/global-pose-utm.txt", delimiter=',')
    ground_truth_utm = ground_truth[start_frame:end_frame+1, 1:3]

    initial_pose = ground_truth[start_frame, 1:]
    initial_movement = np.zeros((1, 6))

    initial_movement[0, 0:3] = initial_pose[0:3]
    initial_movement[0, 3:] = R.from_quat(initial_pose[3:]).as_euler('xyz')

    for i in range(start_frame, end_frame+1):
        frame_str = str(i).zfill(6 - len(str(i)))

        It_start = It_end
        disparity_start = disparity_end

        It_end = imread(f'./input/run1_base_hr/omni_image4/frame{frame_str}.png', as_gray = True)
        Ib_end = imread(f'./input/run1_base_hr/omni_image5/frame{frame_str}.png', as_gray = True)

        It_end, Ib_end = rectify_images(It_end, Ib_end, Kt, Kb, dt, db, imageSize, T)
        disparity_end =  get_disparity(It_end, Ib_end, maxd) #np.load('disparity.npy') 

        if i == int(start_frame):
            movement = initial_movement.T
        else:
            movement = estimate_movement(It_start, It_end, disparity_start, Kt, baseline)
        
        all_movements[i] = movement.T

    all_utm_poses = get_utm_poses(all_movements)

    overhead_file = './input/raster_data/mosaic_utm_20cm.tif'
    plot_path(overhead_file, ground_truth_utm, all_utm_poses)


if __name__ == "__main__":
    run_project(0, 1000)