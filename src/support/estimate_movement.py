import numpy as np
from numpy.linalg import inv
from imageio import imread
from sympy import symbols, diff, sin, cos
from sympy.matrices import Matrix, zeros

from .calculate_jacobians import *
from .rectify_images import rectify_images
from .rpy_from_dcm import rpy_from_dcm
from .dcm_from_rpy import dcm_from_rpy


def estimate_movement(It_1, It_2, disparity, K, baseline):
    # Initial guess...
    T = np.array([[1, 0, 0, 0],
                    [0, 1, 0, -0.02],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    E = epose_from_hpose(T)

    f = (K[0, 0]+K[1, 1])/2
    cx = K[0, 2]
    cy = K[1, 2]
    grad_I2 = np.gradient(It_2)

    iters = 100
    alpha = 0.2

    valid_points = np.transpose(disparity.nonzero())

    for iteration in range(iters):
        R = []
        J = []

        for v, u in valid_points:        
            z_calc = baseline*f/(disparity[v][u])
            u_initial = np.array([[u], [v], [1]])
            p = np.vstack((z_calc * inv(K) @ u_initial, np.array([1])))
            p_trans = T @ p
            u_trans = K @ np.array([[p_trans[0, 0]/p_trans[2, 0]], [p_trans[1, 0]/p_trans[2, 0]], [1]])

            if (0 <= int(round(u_trans[1, 0])) < It_2.shape[0]) and (0 <= int(round(u_trans[0, 0])) < It_2.shape[1]):
                r = It_1[v][u] - It_2[int(round(u_trans[1, 0])), int(round(u_trans[0, 0]))]
            else:
                r = It_1[v][u]

            R.append(r)

            j = get_jacobian(z_calc, u, v, f, cx, cy, baseline, E, grad_I2)

            J.append(j)

        R = np.array(R).reshape(len(R), 1)
        J = np.array(J).reshape(len(J), 6)
        print(E, np.sum(np.square(R))/R.shape[0], iteration)

        E = E + alpha * inv(J.T @ J) @ J.T @ R
        T = hpose_from_epose(E)

    T = hpose_from_epose(E)
    return T


def get_jacobian(z_calc, u1, v1, f, cx, cy, baseline, E, grad_I2):
    x, y, z, roll, pitch, yaw = E

    du_dx, dv_dx = get_dx(z_calc, u1, v1, f, cx, cy, baseline, x, y, z, roll, pitch, yaw)
    du_dy, dv_dy = get_dy(z_calc, u1, v1, f, cx, cy, baseline, x, y, z, roll, pitch, yaw)
    du_dz, dv_dz = get_dz(z_calc, u1, v1, f, cx, cy, baseline, x, y, z, roll, pitch, yaw)
    du_droll, dv_droll = get_droll(z_calc, u1, v1, f, cx, cy, baseline, x, y, z, roll, pitch, yaw)
    du_dpitch, dv_dpitch = get_dpitch(z_calc, u1, v1, f, cx, cy, baseline, x, y, z, roll, pitch, yaw)
    du_dyaw, dv_dyaw = get_dyaw(z_calc, u1, v1, f, cx, cy, baseline, x, y, z, roll, pitch, yaw)

    dI_du = grad_I2[1][v1, u1]
    dI_dv = grad_I2[0][v1, u1]

    dI_dx = dI_du*du_dx + dI_dv*dv_dx
    dI_dy = dI_du*du_dy + dI_dv*dv_dy
    dI_dz = dI_du*du_dz + dI_dv*dv_dz
    dI_droll = dI_du*du_droll + dI_dv*dv_droll
    dI_dpitch = dI_du*du_dpitch + dI_dv*dv_dpitch
    dI_dyaw = dI_du*du_dyaw + dI_dv*dv_dyaw

    j = -np.array([dI_dx, dI_dy, dI_dz, dI_droll, dI_dpitch, dI_dyaw])
    
    return j


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


def get_derivatives():
    z_calc, u1, v1, f, cx, cy, baseline, x, y, z, roll, pitch, yaw = symbols('z_calc, u1, v1, f, cx, cy, baseline, x, y, z, roll, pitch, yaw', real=True)

    K = Matrix([[f, 0, cx], [0, f, cy], [0, 0, 1]])

    T = zeros(4, 4)
    T[0:3, 3] = Matrix([[x], [y], [z]])
    T[3, 3] = 1

    cr = cos(roll)
    sr = sin(roll)
    cp = cos(pitch)
    sp = sin(pitch)
    cy = cos(yaw)
    sy = sin(yaw)

    R = Matrix([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                [  -sp,            cp*sr,            cp*cr]])

    T[0:3, 0:3] = R

    u = Matrix([[u1], [v1], [1]])
    p = zeros(4, 1)
    p[0:3, 0] = z_calc * K.inv() * u
    p[3] = 1
    p_trans = T * p
    u_trans = K * Matrix([[p_trans[0, 0]/p_trans[2, 0]], [p_trans[1, 0]/p_trans[2, 0]], [1]])

    u2 = u_trans[1, 0]
    v2 = u_trans[0, 0]

    print(diff(u2, x))
    print(diff(v2, x))



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
    estimate_movement(It_rect, Ib_rect, disparity, K, 0.12)