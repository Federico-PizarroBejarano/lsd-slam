import numpy as np
from numpy.linalg import inv
from imageio import imread

from .calculate_jacobians import *
from .rectify_images import rectify_images
from .transforms import rpy_from_dcm, dcm_from_rpy, epose_from_hpose, hpose_from_epose


def estimate_movement(It_1, It_2, disparity, K, baseline):
    # Initial guess...
    T = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, -0.1],
                    [0, 0, 0, 1]])
    E = epose_from_hpose(T)

    f = (K[0, 0]+K[1, 1])/2
    cx = K[0, 2]
    cy = K[1, 2]
    grad_I2 = np.gradient(It_2)
    inv_K = inv(K)

    iters = 30
    alpha = 1

    valid_points = np.transpose(disparity.nonzero())
    valid_points = valid_points[np.random.choice(valid_points.shape[0], min(valid_points.shape[0], 5000), replace=False)]
    print("Number of valid points: ", valid_points.shape[0])

    R = np.zeros((valid_points.shape[0], 1))
    J = np.zeros((valid_points.shape[0], 6))

    for iteration in range(iters):

        for point in range(valid_points.shape[0]):
            v, u = valid_points[point]  
            d = disparity[v][u]      
            z_calc = baseline*f/d
            u_initial = np.array([[u], [v], [1]])
            p = np.vstack((z_calc * inv_K @ u_initial, np.array([1])))
            p_trans = T @ p
            u_trans = K @ np.array([[p_trans[0, 0]/p_trans[2, 0]], [p_trans[1, 0]/p_trans[2, 0]], [1]])

            u2, v2 = int(round(u_trans[0, 0])), int(round(u_trans[1, 0]))
            if (0 <= v2 < It_2.shape[0]) and (0 <= u2 < It_2.shape[1]):
                r = It_1[v][u] - It_2[v2, u2]
            else:
                v2, u2 = 0, 0
                r = It_1[v][u]

            R[point, 0] = r

            j = get_jacobian(z_calc, u, v, u2, v2, f, cx, cy, baseline, E, grad_I2)
            J[point, :] = j

        error = np.sum(np.square(R))/R.shape[0]
        print(f'Calculating movement. Error: {error}, Iteration: {iteration}')
        E = E + alpha * inv(J.T @ J) @ J.T @ R
        T = hpose_from_epose(E)

        if error <= 300:
            break

    T = hpose_from_epose(E)
    E = epose_from_hpose(inv(T))
    return E


def get_jacobian(z_calc, u1, v1, u2, v2, f, cx, cy, baseline, E, grad_I2):
    x, y, z, roll, pitch, yaw = E.T[0]

    du_dx, dv_dx = get_dx(z_calc, u1, v1, f, cx, cy, baseline, x, y, z, roll, pitch, yaw)
    du_dy, dv_dy = get_dy(z_calc, u1, v1, f, cx, cy, baseline, x, y, z, roll, pitch, yaw)
    du_dz, dv_dz = get_dz(z_calc, u1, v1, f, cx, cy, baseline, x, y, z, roll, pitch, yaw)
    du_droll, dv_droll = get_droll(z_calc, u1, v1, f, cx, cy, baseline, x, y, z, roll, pitch, yaw)
    du_dpitch, dv_dpitch = get_dpitch(z_calc, u1, v1, f, cx, cy, baseline, x, y, z, roll, pitch, yaw)
    du_dyaw, dv_dyaw = get_dyaw(z_calc, u1, v1, f, cx, cy, baseline, x, y, z, roll, pitch, yaw)

    dI_du = grad_I2[1][v2, u2]
    dI_dv = grad_I2[0][v2, u2]

    dI_dx = dI_du*du_dx + dI_dv*dv_dx
    dI_dy = dI_du*du_dy + dI_dv*dv_dy
    dI_dz = dI_du*du_dz + dI_dv*dv_dz
    dI_droll = dI_du*du_droll + dI_dv*dv_droll
    dI_dpitch = dI_du*du_dpitch + dI_dv*dv_dpitch
    dI_dyaw = dI_du*du_dyaw + dI_dv*dv_dyaw

    j = np.array([dI_dx, dI_dy, dI_dz, dI_droll, dI_dpitch, dI_dyaw])
    
    return j


if __name__ == "__main__":
    It = imread('./input/run1_base_hr/omni_image4/frame000000_2018_09_04_17_19_42_773316.png', as_gray = True)
    Ib = imread('./input/run1_base_hr/omni_image5/frame000000_2018_09_04_17_19_42_773316.png', as_gray = True)
    
    # It = imread('./input/run1_base_hr/omni_image4/frame001577_2018_09_04_17_25_40_946193.png', as_gray = True)
    # Ib = imread('./input/run1_base_hr/omni_image5/frame001577_2018_09_04_17_25_40_946193.png', as_gray = True)

    Kt = np.array([[473.571, 0,  378.17],
            [0,  477.53, 212.577],
            [0,  0,  1]])
    Kb = np.array([[473.368, 0,  371.65],
            [0,  477.558, 204.79],
            [0,  0,  1]])
    dt = np.array([-0.333605,0.159377,6.11251e-05,4.90177e-05,-0.0460505])
    db = np.array([-0.3355,0.162877,4.34759e-05,2.72184e-05,-0.0472616])

    imageSize = (752, 480)
    T = np.array([0, 0.12, 0])

    It_rect, Ib_rect, K = rectify_images(It, Ib, Kt, Kb, dt, db, imageSize, T)
    disparity = np.load('disparity.npy')
    disparity[disparity > 25] = 0
    estimate_movement(It_rect, Ib_rect, disparity, K, 0.12)