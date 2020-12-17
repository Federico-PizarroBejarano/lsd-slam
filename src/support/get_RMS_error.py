import math
import numpy as np

def get_RMS_error(ground_truth, calculated_poses, timestamps):
    error = 0
    closest_path = np.zeros((len(calculated_poses), 2))

    for i in range(len(calculated_poses)):
        closest_truth = find_closest_pose(ground_truth, timestamps[i])
        closest_path[i, :] = closest_truth

        print(f'Frame #: {i}, True Pose: {closest_truth}, Calculated Pose: {calculated_poses[i]}')
        error += np.linalg.norm(calculated_poses[i] - closest_truth)**2

    return math.sqrt(error/len(timestamps)), closest_path

def find_closest_pose(ground_truth, time_stamp):
    idx = np.searchsorted(ground_truth[:, 0], time_stamp, side="left")
    pose0 = ground_truth[idx-1, 1:]
    pose1 = ground_truth[idx, 1:]
    t0 = ground_truth[idx-1, 0]
    t1 = ground_truth[idx, 0]
    interpolated_pose = pose0 + (time_stamp - t0)*(pose1 - pose0)/(t1 - t0)
    return interpolated_pose
