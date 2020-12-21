from pathlib import Path
import argparse
import sys

import os
import time
from datetime import datetime, timedelta
import numpy as np
from imageio import imread
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from support.get_disparity import get_disparity
from support.rectify_images import rectify_images
from support.estimate_movement import estimate_movement
from support.plot_path import plot_path
from support.get_utm_poses import get_utm_poses
from support.get_RMS_error import get_RMS_error, find_closest_pose
from support.file_management import get_filename, get_unix_timestamp
from support.transforms import epose_from_hpose
from support.check_outlier import check_outlier


def run_project(input_dir, output_dir):
    # Get start time to check runtime
    start_time = time.time()

    # Setting the start and end frames
    start_frame = 1750
    end_frame = 1751

    # Get all filenames in omni_images4 folder
    files = os.listdir(f'{input_dir}/run1_base_hr/omni_image4')

    # Get all ground truth poses and set initial movement
    ground_truth = np.genfromtxt(f"{input_dir}/run1_base_hr/global-pose-utm.txt", delimiter=',')
    ground_truth_utm = ground_truth[:, 0:3]

    initial_pose = find_closest_pose(ground_truth, get_unix_timestamp(get_filename(files, str(start_frame).zfill(6))))

    initial_movement = np.zeros((6, 1))
    initial_movement[0:3, :] = np.reshape(initial_pose[0:3].T, (3, 1))
    initial_movement[0:3, :] = np.reshape(R.from_quat(initial_pose[3:]).as_euler('xyz'), (3, 1))

    # Initialize arrays
    It_end = []
    disparity_end = []
    all_movements = []
    timestamps = []

    # Set distance between stereo cameras
    baseline = 0.12

    # Loop through every frame from start_frame to end_frame
    for frame in range(start_frame, end_frame+1):
        # Save previous rectified images and disparity values
        It_start = It_end
        disparity_start = disparity_end

        # Read in new images
        filename = get_filename(files, frame)
        It_end = imread(f'{input_dir}/run1_base_hr/omni_image4/{filename}', as_gray = True)
        Ib_end = imread(f'{input_dir}/run1_base_hr/omni_image5/{filename}', as_gray = True)

        # Rectify new images and calculate disparity
        It_end, Ib_end, K_rect = rectify_images(It_end, Ib_end)
        disparity_end =  get_disparity(It_end, Ib_end) 
        
        # Saving disparity as an image
        plt.imshow(-disparity_end, cmap='gray')
        plt.savefig(f'{output_dir}/disparity_frame{frame}.png')

        # If it is the first frame, set first movement to inital_movement
        if frame == int(start_frame):
            movement = initial_movement
            all_movements.append(movement.T[0])
            timestamps.append(get_unix_timestamp(filename))
        else:
            # Estimate the movement from the two top images and the disparity
            movement, error = estimate_movement(It_start, It_end, disparity_start, K_rect, baseline)
        
            if check_outlier(movement, error) == False:
                all_movements.append(movement.T[0])
                timestamps.append(get_unix_timestamp(filename))
        
        print(f'Frame: {frame}')        

    # Calculate the utm coordinates given the movements
    calculated_utm_poses = get_utm_poses(np.array(all_movements))

    # Calculate the RMS error
    rms_error, closest_path = get_RMS_error(ground_truth_utm, calculated_utm_poses, timestamps)
    print('RMS Error: ', rms_error)
    print('Runtime: ', time.time()-start_time)

    # Plot final calculated path
    plot_path(input_dir, output_dir, ground_truth_utm[:, 1:], closest_path, calculated_utm_poses)


parser = argparse.ArgumentParser(description='ROB501 Final Project.')
parser.add_argument('--input_dir', dest='input_dir', type=str, default="./input",
                    help='Input Directory that contains all required rover data')
parser.add_argument('--output_dir', dest='output_dir', type=str, default="./output",
                    help='Output directory where all outputs will be stored.')


if __name__ == "__main__":
    # Parse command line arguments
    args = parser.parse_args()

    # Run the project code
    run_project(args.input_dir, args.output_dir)
