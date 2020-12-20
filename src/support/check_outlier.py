import numpy as np
from scipy.spatial.transform import Rotation as R

def check_outlier(movement):
    if np.linalg.norm(movement[0:3]) > 0.4:
        print("movement")
        return True
    
    axis_of_rot = R.from_euler('xyz', movement[3:].T[0]).as_rotvec()
    if np.linalg.norm(axis_of_rot) > 0.2:
        print('rot')
        return True
    
    return False
