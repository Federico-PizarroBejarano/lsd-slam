import numpy as np

from .transforms import hpose_from_epose, epose_from_hpose, TR_C, TC_R

def get_utm_poses(all_movements):
    utm_poses = np.zeros((all_movements.shape[0],2))
    initial_movement = np.reshape(all_movements[0], (6,1))
    utm_poses[0, :] = initial_movement[0:2, 0].T

    current_transform = hpose_from_epose(initial_movement)

    for i in range(1, all_movements.shape[0]):
        small_movement = hpose_from_epose(all_movements[i, :])
        current_transform = current_transform @ TR_C() @ small_movement @ TC_R()
        utm_poses[i] = current_transform[0:2, 3].T

    return utm_poses
