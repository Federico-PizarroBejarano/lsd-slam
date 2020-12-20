import numpy as np

from .transforms import hpose_from_epose, epose_from_hpose, TR_C, TC_R

def get_utm_poses(all_movements):
    """
    Calculated the utm coordinates of the rover's body frame given a numpy array of
    transformations. The first transformation moves the rover from the site frame (at the edge 
    of UTM 18t) to the starting position and orientation of rover. The rest are transforms from
    one frame to the next in the frame of onmi-camera 4. 

    Parameters
    ----------
    all_movements (np.ndarray): a numpy array where each row contains the transformation the
        rover undergoes between consecutive frames
    
    Returns
    -------
    utm_poses (np.ndarray): the calculated utm coordinates
    """

    utm_poses = np.zeros((all_movements.shape[0],2))

    # Find initial movement and set first utm coordinate to initial rover position
    initial_movement = np.reshape(all_movements[0], (6,1))
    utm_poses[0, :] = initial_movement[0:2, 0].T

    current_transform = hpose_from_epose(initial_movement)

    # Iterate through remaining transforms
    for i in range(1, all_movements.shape[0]):
        small_movement = hpose_from_epose(all_movements[i, :])
        current_transform = current_transform @ TR_C() @ small_movement @ TC_R()
        utm_poses[i] = current_transform[0:2, 3].T

    return utm_poses
