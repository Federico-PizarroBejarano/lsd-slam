import numpy as np
from scipy.spatial.transform import Rotation as R

def check_outlier(movement, error):
    """
    Checks if a calculated movement transform is an outlier. 
    It does this by checking if the photometric error, translation, or rotation is too large. 

    Parameters
    ----------
    movement (np.ndarray): a (6,1) numpy array representing the twist (x, y, z, roll, pitch, yaw)
        between two frames 
    error (float): the photometric error when using this twist
    
    Returns
    -------
    outlier_status (boolean): True if it is an outlier, False otherwise.
    """

    error_threshold = 2000
    translation_threshold = 0.5
    rotation_threshold = 0.4

    if error > error_threshold:
        return True

    if np.linalg.norm(movement[0:3]) > translation_threshold:
        return True
    
    # Use axis-angle representation to check magnitude of rotation
    axis_of_rot = R.from_euler('xyz', movement[3:].T[0]).as_rotvec()
    if np.linalg.norm(axis_of_rot) > rotation_threshold:
        return True
    
    return False
