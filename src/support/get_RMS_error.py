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
    return (ground_truth[idx-1, 1:] + ground_truth[idx, 1:])/2
