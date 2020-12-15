import numpy as np
from scipy.spatial.transform import Rotation as R

from .dcm_from_rpy import dcm_from_rpy 

def get_utm_poses(all_movements):
    utm_poses = np.zeros((all_movements.shape[0],2))
    utm_poses[0, :] = all_movements[0, 0:2]

    current_transform = hpose_from_epose(all_movements[0, :])

    for i in range(1, all_movements.shape[0]):
        T = hpose_from_epose(all_movements[i, :])
        current_transform = current_transform @ T
        utm_poses[i+1] = current_transform[0:2, 3].T
    
    return utm_poses


def hpose_from_epose(E):
    """Covert x, y, z, roll, pitch, yaw to 4x4 homogeneous pose matrix."""
    T = np.zeros((4, 4))
    T[0:3, 0:3] = dcm_from_rpy(E[3:6])
    T[0:3, 3] = np.reshape(E[0:3], (3,))
    T[3, 3] = 1
  
    return T