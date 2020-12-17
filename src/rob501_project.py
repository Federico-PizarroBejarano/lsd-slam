import os
import time
from datetime import datetime, timedelta
import numpy as np
from imageio import imread
from scipy.spatial.transform import Rotation as R

from .support.get_disparity import get_disparity
from .support.rectify_images import rectify_images
from .support.estimate_movement import estimate_movement
from .support.plot_path import plot_path
from .support.get_utm_poses import get_utm_poses
from .support.get_RMS_error import get_RMS_error
from .support.convert_time import convert_date_string_to_unix_seconds


def run_project(start_frame, end_frame):
    start_time = time.time()
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

    ground_truth = np.genfromtxt("./input/run1_base_hr/global-pose-utm.txt", delimiter=',')
    ground_truth_utm = ground_truth[:, 0:3]

    initial_pose = ground_truth[start_frame, 1:]
    initial_movement = np.zeros((1, 6))

    initial_movement[0, 0:3] = initial_pose[0:3]
    initial_movement[0, 3:] = R.from_quat(initial_pose[3:]).as_euler('xyz')

    files = os.listdir('./input/run1_base_hr/omni_image4')
    all_movements = []
    timestamps = []

    It_end = []
    disparity_end = []

    for i in range(start_frame, end_frame+1):
        frame_str = str(i).zfill(6)

        It_start = It_end
        disparity_start = disparity_end

        filename = get_filename(files, frame_str)
        It_end = imread(f'./input/run1_base_hr/omni_image4/{filename}', as_gray = True)
        Ib_end = imread(f'./input/run1_base_hr/omni_image5/{filename}', as_gray = True)

        It_end, Ib_end, K_rect = rectify_images(It_end, Ib_end, Kt, Kb, dt, db, imageSize, T)
        disparity_end =  get_disparity(It_end, Ib_end, maxd) 
        # disparity_end = np.load(f'disparity_{i}.npy')
        np.save(f'disparity_{i}', disparity_end)

        if i == int(start_frame):
            movement = initial_movement.T
        else:
            movement = estimate_movement(It_start, It_end, disparity_start, K_rect, baseline)

        all_movements.append(movement.T[0])
        print('Movement: ', movement, 'Time: ', time.time() - start_time)
        timestamps.append(get_timestamp(filename))

    print('All Movement: ', all_movements, 'Time: ', time.time() - start_time)
    all_utm_poses = get_utm_poses(np.array(all_movements))
    print('All UTM Poses: ', all_utm_poses, 'Time: ', time.time() - start_time)

    RMS_error, closest_path = get_RMS_error(ground_truth_utm, all_utm_poses, timestamps)
    print('RMS Error: ', RMS_error)
    print('Runtime: ', time.time()-start_time)

    plot_path(closest_path, all_utm_poses)


def get_filename(files, frame):
    corresponding_files = [image for image in files if f'frame{frame}' in image]
    return corresponding_files[0]

def get_timestamp(filename):
    time_str = filename.split('frame')[1][7:-4]
    epoch_time = convert_date_string_to_unix_seconds(time_str)
    return epoch_time


if __name__ == "__main__":
    run_project(70, 80)
