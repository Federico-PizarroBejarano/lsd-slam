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
from .support.get_RMS_error import get_RMS_error, find_closest_pose
from .support.convert_time import convert_date_string_to_unix_seconds
from .support.transforms import epose_from_hpose, hpose_from_epose
from .support.check_outlier import check_outlier


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
    files = os.listdir('./input/run1_base_hr/omni_image4')

    ground_truth = np.genfromtxt("./input/run1_base_hr/global-pose-utm.txt", delimiter=',')
    ground_truth_utm = ground_truth[:, 0:3]

    initial_pose = find_closest_pose(ground_truth, get_timestamp(get_filename(files, str(start_frame).zfill(6))))

    TSR = np.eye(4)
    TSR[0:3, 3] = initial_pose[0:3].T
    TSR[0:3, 0:3] = R.from_quat(initial_pose[3:]).as_matrix()
    initial_movement = epose_from_hpose(TSR)

    all_movements = []
    timestamps = []
    error_threshold = 1500

    It_end = []
    disparity_end = []
    movement_dictionary = np.load('./output/movements3.npy', allow_pickle = True).item()

    for i in range(start_frame, end_frame+1):
        frame_str = str(i).zfill(6)

        It_start = It_end
        disparity_start = disparity_end

        filename = get_filename(files, frame_str)
        It_end = imread(f'./input/run1_base_hr/omni_image4/{filename}', as_gray = True)
        Ib_end = imread(f'./input/run1_base_hr/omni_image5/{filename}', as_gray = True)

        It_end, Ib_end, K_rect = rectify_images(It_end, Ib_end, Kt, Kb, dt, db, imageSize, T)
        disparity_end =  get_disparity(It_end, Ib_end, maxd) 
        # disparity_end = np.load(f'./output/disparity_{i}.npy')
        np.save(f'./output/disparity_{i}', disparity_end)

        if i == int(start_frame):
            movement = initial_movement
            all_movements.append(movement.T[0])
            timestamps.append(get_timestamp(filename))
        else:
            if i in movement_dictionary:
                movement, error = movement_dictionary[i] # = np.array([[0, 0, 0.1, 0, 0, 0]]).T#
            else:
                movement, error = estimate_movement(It_start, It_end, disparity_start, K_rect, baseline)
        
            # if check_outlier(movement) == False and error <= error_threshold:
            all_movements.append(movement.T[0])
            timestamps.append(get_timestamp(filename))
            # else:
            #     print('outlier', check_outlier(movement), error)
            movement_dictionary[i] = (movement, error)
            np.save('./output/movements3.npy', movement_dictionary)
        
        print(f'Frame: {i}')        

    print('All Movement: ', all_movements, 'Time: ', time.time() - start_time)
    all_utm_poses = get_utm_poses(np.array(all_movements))
    print('All UTM Poses: ', all_utm_poses, 'Time: ', time.time() - start_time)

    RMS_error, closest_path = get_RMS_error(ground_truth_utm, all_utm_poses, timestamps)
    print('RMS Error: ', RMS_error)
    print('Runtime: ', time.time()-start_time)

    plot_path(ground_truth_utm[:, 1:], closest_path, all_utm_poses)


def get_filename(files, frame):
    corresponding_files = [image for image in files if f'frame{frame}' in image]
    return corresponding_files[0]

def get_timestamp(filename):
    time_str = filename.split('frame')[1][7:-4]
    epoch_time = convert_date_string_to_unix_seconds(time_str)
    return epoch_time


if __name__ == "__main__":
    run_project(1844, 1900)
