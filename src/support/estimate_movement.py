import numpy as np
from numpy.linalg import inv
from imageio import imread

from .calculate_jacobians import get_dx, get_dy, get_dz, get_droll, get_dpitch, get_dyaw
from .transforms import rpy_from_dcm, dcm_from_rpy, epose_from_hpose, hpose_from_epose
from .rectify_images import rectify_images
from .get_disparity import get_disparity


def estimate_movement(It_1, It_2, disparity, K, baseline):
    """
    Estimate the twist from It_2 to It_1 given the two images and the disparity map of the first image.  

    Parameters
    ----------
    It_1 (np.ndarray): First top image (template image), m x n pixel np.array, greyscale.
    It_2 (np.ndarray): Second top image, m x n pixel np.array, greyscale.
    disparity (np.ndarray): Disparity image (map) as np.array, same size as It_1 and It_2.
    K (np.ndarray): the instrinic camera matrix
    baseline (float): the distance between the two stereo cameras used to create the disparity map
    
    Returns
    -------
    E (np.ndarray): a (6,1) numpy array representing the twist (x, y, z, roll, pitch, yaw) between two frames 
    error (float): the photometric error when using this twist
    """

    # Initial guess of moving forward 0.1m
    T = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, -0.1],
                    [0, 0, 0, 1]])
    E = epose_from_hpose(T)

    # Setting up useful variables
    f = (K[0, 0]+K[1, 1])/2
    cu = K[0, 2]
    cv = K[1, 2]
    grad_I2 = np.gradient(It_2)
    inv_K = inv(K)

    # Removing disparity values near edge to guarantee pixel is in both images
    border = 20
    width = It_1.shape[1]
    height = It_1.shape[0]
    edge_mask = np.zeros(It_1.shape)
    edge_mask[border:height-border, border:width-border] = np.ones((height - border*2, width - border*2))
    disparity[edge_mask == 0] = 0

    # Get array of coordinates of valid disparity points
    valid_points = np.transpose(disparity.nonzero())

    # Randomly sample (without replacement) num_point number of points from valid points to reduce runtime
    num_points = 15000
    valid_points = valid_points[np.random.choice(valid_points.shape[0], min(valid_points.shape[0], num_points), replace=False)]

    # Set up initial values
    R = np.zeros((valid_points.shape[0], 1))
    J = np.zeros((valid_points.shape[0], 6))
    error = 0
    no_change_counter = 0

    iters = 25

    for iteration in range(iters):
        previous_error = error

        for point in range(valid_points.shape[0]):
            v, u = valid_points[point]   
            d = disparity[v][u]
            z_calc = baseline*f/d

            # Calculate residuals related to point
            r, u2, v2 = get_residual(z_calc, u, v, f, baseline, K, inv_K, T, It_1, It_2)
            R[point, 0] = r

            # Calculate jacobian of residual at point
            j = get_jacobian(z_calc, u, v, u2, v2, f, cu, cv, E, grad_I2)
            J[point, :] = j

        # Calculate average squared residual
        error = np.sum(np.square(R))/R.shape[0]
        print(f'Calculating movement. Error: {error}, Iteration: {iteration}')

        # Update twist
        E = E + inv(J.T @ J) @ J.T @ R
        T = hpose_from_epose(E)

        # Terminate if error sufficiently low or very little change in error for too many iterations
        if error <= 200:
            break

        if abs(previous_error - error) < 10:
            no_change_counter += 1
        else:
            no_change_counter = 0

        if no_change_counter >= 4:
            break

    # Invert transform to get transform fro It_1 to It_2 (rather than the other way around)
    T = hpose_from_epose(E)
    E = epose_from_hpose(inv(T))
    return E, error


def get_residual(z_calc, u, v, f, baseline, K, inv_K, T, It_1, It_2):
    """
    Calculate the photometric residual at a point given two images and the disparity map

    Parameters
    ----------
    z_calc (float): Calculated depth of point from disparity
    u (int): u coordinate of point in the template (first) image
    v (int): v coordinate of point in the template (first) image
    f (float): focal length of camera
    baseline (float): the distance between the two stereo cameras used to create the disparity map
    K (np.ndarray): the instrinic camera matrix
    inv_K (np.ndarray): the inverse of K
    T (np.ndarray): a 4x4 numpy array representing the current estimate for the homogenous transform
    It_1 (np.ndarray): First top image (template image), m x n pixel np.array, greyscale.
    It_2 (np.ndarray): Second top image, m x n pixel np.array, greyscale.
    
    Returns
    -------
    residual (float): the photometric error residual at that point (u, v)
    """

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
    
    return r, u2, v2


def get_jacobian(z_calc, u1, v1, u2, v2, f, cu, cv, E, grad_I2):
    """
    Get the jacobian of a residual with respect to twist at a specific point (u1, v1)

    Parameters
    ----------
    z_calc (float): Calculated depth of point from disparity
    u1 (int): u coordinate of point in the template image
    v1 (int): v coordinate of point in the template image
    u2 (int): u coordinate of point in the second image
    v2 (int): v coordinate of point in the second image
    f (float): focal length of camera
    cu (float): the u coordinate of the principle point of the camera
    cv (float): the v coordinate of the principle point of the camera
    E (np.ndarray): a 6x1 numpy array representing the twist (x, y, z, roll, pitch, yaw)
    grad_I2 (np.ndarray): the gradient of the second image found using np.gradient
    
    Returns
    -------
    jacobian (np.ndarray): a 1x6 numpy array representing the jacobian of the residual w.r.t the twist
    """

    # Getting twist variables
    x, y, z, roll, pitch, yaw = E.T[0]

    # Calculating derivatives of u and v
    du_dx, dv_dx = get_dx(z_calc, u1, v1, f, cu, cv, x, y, z, roll, pitch, yaw)
    du_dy, dv_dy = get_dy(z_calc, u1, v1, f, cu, cv, x, y, z, roll, pitch, yaw)
    du_dz, dv_dz = get_dz(z_calc, u1, v1, f, cu, cv, x, y, z, roll, pitch, yaw)
    du_droll, dv_droll = get_droll(z_calc, u1, v1, f, cu, cv, x, y, z, roll, pitch, yaw)
    du_dpitch, dv_dpitch = get_dpitch(z_calc, u1, v1, f, cu, cv, x, y, z, roll, pitch, yaw)
    du_dyaw, dv_dyaw = get_dyaw(z_calc, u1, v1, f, cu, cv, x, y, z, roll, pitch, yaw)

    # Getting gradient values at point (u, v) in second image
    dI_du = grad_I2[1][v2, u2]
    dI_dv = grad_I2[0][v2, u2]

    # Calculating jacobian
    dI_dx = dI_du*du_dx + dI_dv*dv_dx
    dI_dy = dI_du*du_dy + dI_dv*dv_dy
    dI_dz = dI_du*du_dz + dI_dv*dv_dz
    dI_droll = dI_du*du_droll + dI_dv*dv_droll
    dI_dpitch = dI_du*du_dpitch + dI_dv*dv_dpitch
    dI_dyaw = dI_du*du_dyaw + dI_dv*dv_dyaw

    j = np.array([dI_dx, dI_dy, dI_dz, dI_droll, dI_dpitch, dI_dyaw])
    
    return j


if __name__ == "__main__":
    # Read in two images
    It = imread('./input/run1_base_hr/omni_image4/frame000000_2018_09_04_17_19_42_773316.png', as_gray = True)
    Ib = imread('./input/run1_base_hr/omni_image5/frame000000_2018_09_04_17_19_42_773316.png', as_gray = True)

    # Set up values needed for rectification
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
    baseline = 0.12

    # Rectify the images and calculate disparity
    It_rect, Ib_rect, K = rectify_images(It, Ib, Kt, Kb, dt, db, imageSize, T)
    disparity = get_disparity(It_rect, Ib_rect)    

    # Estimate transform between top and bottom image
    E, error = estimate_movement(It_rect, Ib_rect, disparity, K, baseline)
    print(E, error)
