import math
import numpy as np

def get_RMS_error(ground_truth, calculated_poses, timestamps):
    """
    Calculates the RMS error between calculated utm poses and the ground truth.
    The ground truth needs to be interpolated since it is not provided for the exact 
    time of each frame. 

    Parameters
    ----------
    ground_truth (np.ndarray): a numpy array where each row contains the timestamp and the utm
        coordinates of the rover's body frame
    calculated_poses (np.ndarray): the calculated utm coordinates of the rover's body frame
    timestamps (list): a list of the unix timestamps corresponding to each calculated pose
    
    Returns
    -------
    rms_error (float): the root-mean-squared error of the calculated utm positions
    closest_path (np.ndarray): an array of the interpolated ground truth poses at the time
        of each frame, used for comparison.
    """

    error = 0
    closest_path = np.zeros((len(calculated_poses), 2))

    for i in range(len(calculated_poses)):
        # Interpolate the ground truth at this timestamp
        closest_truth = find_closest_pose(ground_truth, timestamps[i])
        closest_path[i, :] = closest_truth

        # Accumulate squared error
        error += np.linalg.norm(calculated_poses[i] - closest_truth)**2

    # Calculate the root mean squared error
    rms_error = math.sqrt(error/len(timestamps))

    return rms_error, closest_path


def find_closest_pose(ground_truth, timestamp):
    """
    Linearly interpolates the utm coordinates closest to the provided timestamp

    Parameters
    ----------
    ground_truth (np.ndarray): a numpy array where each row contains the timestamp and the utm
        coordinates of the rover's body frame
    timestamp (float): a unix timestamp in seconds
    
    Returns
    -------
    interpolated_pose (np.ndarray): the interpolated utm coordinates
    """

    # Find the index where this timestamp would be inserted into the ground truth
    idx = np.searchsorted(ground_truth[:, 0], timestamp, side="left")

    # Perform linear interpolation between values at idx-1 and idx
    pose0 = ground_truth[idx-1, 1:]
    pose1 = ground_truth[idx, 1:]
    t0 = ground_truth[idx-1, 0]
    t1 = ground_truth[idx, 0]

    interpolated_pose = pose0 + (timestamp - t0)*(pose1 - pose0)/(t1 - t0)

    return interpolated_pose
