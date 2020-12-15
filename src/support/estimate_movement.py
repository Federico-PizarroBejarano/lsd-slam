import numpy as np
from numpy.linalg import inv

from rpy_from_dcm import rpy_from_dcm
from dcm_from_rpy import dcm_from_rpy

def epose_from_hpose(T):
    """Covert 4x4 homogeneous pose matrix to x, y, z, roll, pitch, yaw."""
    E = np.zeros((6, 1))
    E[0:3] = np.reshape(T[0:3, 3], (3, 1))
    E[3:6] = rpy_from_dcm(T[0:3, 0:3])
  
    return E

def hpose_from_epose(E):
    """Covert x, y, z, roll, pitch, yaw to 4x4 homogeneous pose matrix."""
    T = np.zeros((4, 4))
    T[0:3, 0:3] = dcm_from_rpy(E[3:6])
    T[0:3, 3] = np.reshape(E[0:3], (3,))
    T[3, 3] = 1
  
    return T

def estimate_movement(It_1, It_2, disparity, K, baseline):
    # Initial guess...
    T = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0.10],
                    [0, 0, 0, 1]])
    C = T[:3, :3]
    rpy = rpy_from_dcm(C).reshape(3, 1)

    f = (K[0, 0]+K[1, 1])/2

    iters = 100
    # Iterate.
    for j in range(iters):
        R = []
        J = []

        for y in range(disparity.shape[0]):
            for x in range(disparity.shape[1]):
                if disparity[y][x] == 0:
                    continue
            
                z = baseline*f/(disparity[y][x])
                u = np.array([[x], [y], [1]])
                p = np.vstack((z * inv(K) @ u, np.array([1])))
                p_trans = T @ p
                u_trans = K @ np.array([[p_trans[0, 0]/p_trans[2, 0]], [p_trans[1, 0]/p_trans[2, 0]], [1]])
                r = It_1[y][x] - It_2[int(round(u_trans[1, 0])), int(round(u_trans[0, 0]))]

                R.append(r)

                j = 0

                J.append(j)

        R = np.array(R)
        J = np.array(J)

        theta = theta + inv(J.T @ J) @ J.T @ R
        rpy = theta[0:3].reshape(3, 1)
        C = dcm_from_rpy(rpy)
        t = theta[3:6].reshape(3, 1)

    T = np.vstack((np.hstack((C, t)), np.array([[0, 0, 0, 1]])))
    return T