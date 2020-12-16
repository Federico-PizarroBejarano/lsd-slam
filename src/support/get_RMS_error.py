import math
import numpy as np

def get_RMS_error(ground_truth, calculated_pose, timestamps):
    error = 0

    for i in range(len(calculated_pose)):
        closest_truth = find_closest_pose(ground_truth, timestamps[i])
        error += np.linalg.norm(calculated_pose[0:3] - closest_truth[0:3])**2
    
    return math.sqrt(error/len(timestamps))

def find_closest_pose(ground_truth, time_stamp):
    idx = np.searchsorted(ground_truth[:, 0], time_stamp, side="left")
    pose0 = ground_truth[idx-1, 1:]
    pose1 = ground_truth[idx, 1:]
    t0 = ground_truth[idx-1, 0]
    t1 = ground_truth[idx, 0]
    interpolated_pose = pose0 + (time_stamp - t0)*(pose1 - pose0)/(t1 - t0)
    return interpolated_pose
