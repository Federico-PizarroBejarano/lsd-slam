"""Calculate Jacobians

Has functions returning the derivative of u and v (image plane coordinates) with respect to
x, y, z, roll, pitch, and yaw (the twist parameters). Also includes a function 
"calculate_symbolic_derivatives" that uses SymPy to calculate the form of the derivatives, 
and tests the derivative functions. 

The input to all the derivative functions are:
    z_calc (float): Calculated depth of point from disparity
    u1 (int): u coordinate of point in the template image
    v1 (int): v coordinate of point in the template image
    f (float): focal length of camera
    cu (float): the u coordinate of the principle point of the camera
    cv (float): the v coordinate of the principle point of the camera
    x (float): the current estimate for translation along the x-axis
    y (float): the current estimate for translation along the y-axis
    z (float): the current estimate for translation along the z-axis
    roll (float): the current estimate for roll angle
    pitch (float): the current estimate for pitch angle
    yaw (float): the current estimate for yaw angle
"""

from math import sin, cos
import sympy
from sympy.matrices import Matrix, zeros

def get_dx(z_calc, u1, v1, f, cu, cv, x, y, z, roll, pitch, yaw):
    """
    Calculates the derivatives of u and v with respect to x

    Parameters
    ----------
    z_calc, u1, v1, f, cu, cv, x, y, z, roll, pitch, yaw, as explained in module docstring
    
    Returns
    -------
    du_dx (float): the derivative of u w.r.t. x
    dv_dx (float): the derivative of v w.r.t. x
    """

    du = f/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))
    dv = 0

    return du, dv


def get_dy(z_calc, u1, v1, f, cu, cv, x, y, z, roll, pitch, yaw):
    """
    Calculates the derivatives of u and v with respect to y

    Parameters
    ----------
    z_calc, u1, v1, f, cu, cv, x, y, z, roll, pitch, yaw, as explained in module docstring
    
    Returns
    -------
    du_dy (float): the derivative of u w.r.t. y
    dv_dy (float): the derivative of v w.r.t. y
    """

    du = 0
    dv = f/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))

    return du, dv


def get_dz(z_calc, u1, v1, f, cu, cv, x, y, z, roll, pitch, yaw):
    """
    Calculates the derivatives of u and v with respect to z

    Parameters
    ----------
    z_calc, u1, v1, f, cu, cv, x, y, z, roll, pitch, yaw, as explained in module docstring
    
    Returns
    -------
    du_dz (float): the derivative of u w.r.t. z
    dv_dz (float): the derivative of v w.r.t. z
    """

    du = -f*(x + z_calc*(sin(pitch)*cos(roll)*cos(yaw) + sin(roll)*sin(yaw)) + (-cu*z_calc/f + u1*z_calc/f)*cos(pitch)*cos(yaw) + (-cv*z_calc/f + v1*z_calc/f)*(sin(pitch)*sin(roll)*cos(yaw) - sin(yaw)*cos(roll)))/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))**2
    dv = -f*(y + z_calc*(sin(pitch)*sin(yaw)*cos(roll) - sin(roll)*cos(yaw)) + (-cu*z_calc/f + u1*z_calc/f)*sin(yaw)*cos(pitch) + (-cv*z_calc/f + v1*z_calc/f)*(sin(pitch)*sin(roll)*sin(yaw) + cos(roll)*cos(yaw)))/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))**2

    return du, dv
    

def get_droll(z_calc, u1, v1, f, cu, cv, x, y, z, roll, pitch, yaw):
    """
    Calculates the derivatives of u and v with respect to roll

    Parameters
    ----------
    z_calc, u1, v1, f, cu, cv, x, y, z, roll, pitch, yaw, as explained in module docstring
    
    Returns
    -------
    du_droll (float): the derivative of u w.r.t. roll
    dv_droll (float): the derivative of v w.r.t. roll
    """

    du = f*(z_calc*(-sin(pitch)*sin(roll)*cos(yaw) + sin(yaw)*cos(roll)) + (-cv*z_calc/f + v1*z_calc/f)*(sin(pitch)*cos(roll)*cos(yaw) + sin(roll)*sin(yaw)))/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch)) + f*(z_calc*sin(roll)*cos(pitch) - (-cv*z_calc/f + v1*z_calc/f)*cos(pitch)*cos(roll))*(x + z_calc*(sin(pitch)*cos(roll)*cos(yaw) + sin(roll)*sin(yaw)) + (-cu*z_calc/f + u1*z_calc/f)*cos(pitch)*cos(yaw) + (-cv*z_calc/f + v1*z_calc/f)*(sin(pitch)*sin(roll)*cos(yaw) - sin(yaw)*cos(roll)))/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))**2
    dv = f*(z_calc*(-sin(pitch)*sin(roll)*sin(yaw) - cos(roll)*cos(yaw)) + (-cv*z_calc/f + v1*z_calc/f)*(sin(pitch)*sin(yaw)*cos(roll) - sin(roll)*cos(yaw)))/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch)) + f*(z_calc*sin(roll)*cos(pitch) - (-cv*z_calc/f + v1*z_calc/f)*cos(pitch)*cos(roll))*(y + z_calc*(sin(pitch)*sin(yaw)*cos(roll) - sin(roll)*cos(yaw)) + (-cu*z_calc/f + u1*z_calc/f)*sin(yaw)*cos(pitch) + (-cv*z_calc/f + v1*z_calc/f)*(sin(pitch)*sin(roll)*sin(yaw) + cos(roll)*cos(yaw)))/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))**2

    return du, dv
    

def get_dpitch(z_calc, u1, v1, f, cu, cv, x, y, z, roll, pitch, yaw):
    """
    Calculates the derivatives of u and v with respect to pitch

    Parameters
    ----------
    z_calc, u1, v1, f, cu, cv, x, y, z, roll, pitch, yaw, as explained in module docstring
    
    Returns
    -------
    du_dpitch (float): the derivative of u w.r.t. pitch
    dv_dpitch (float): the derivative of v w.r.t. pitch
    """

    du = f*(z_calc*sin(pitch)*cos(roll) - (cu*z_calc/f - u1*z_calc/f)*cos(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(pitch)*sin(roll))*(x + z_calc*(sin(pitch)*cos(roll)*cos(yaw) + sin(roll)*sin(yaw)) + (-cu*z_calc/f + u1*z_calc/f)*cos(pitch)*cos(yaw) + (-cv*z_calc/f + v1*z_calc/f)*(sin(pitch)*sin(roll)*cos(yaw) - sin(yaw)*cos(roll)))/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))**2 + f*(z_calc*cos(pitch)*cos(roll)*cos(yaw) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch)*cos(yaw) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch)*cos(yaw))/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))
    dv = f*(z_calc*sin(pitch)*cos(roll) - (cu*z_calc/f - u1*z_calc/f)*cos(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(pitch)*sin(roll))*(y + z_calc*(sin(pitch)*sin(yaw)*cos(roll) - sin(roll)*cos(yaw)) + (-cu*z_calc/f + u1*z_calc/f)*sin(yaw)*cos(pitch) + (-cv*z_calc/f + v1*z_calc/f)*(sin(pitch)*sin(roll)*sin(yaw) + cos(roll)*cos(yaw)))/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))**2 + f*(z_calc*sin(yaw)*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch)*sin(yaw) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*sin(yaw)*cos(pitch))/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))
    
    return du, dv
    

def get_dyaw(z_calc, u1, v1, f, cu, cv, x, y, z, roll, pitch, yaw):
    """
    Calculates the derivatives of u and v with respect to yaw

    Parameters
    ----------
    z_calc, u1, v1, f, cu, cv, x, y, z, roll, pitch, yaw, as explained in module docstring
    
    Returns
    -------
    du_dyaw (float): the derivative of u w.r.t. yaw
    dv_dyaw (float): the derivative of v w.r.t. yaw
    """

    du = f*(z_calc*(-sin(pitch)*sin(yaw)*cos(roll) + sin(roll)*cos(yaw)) - (-cu*z_calc/f + u1*z_calc/f)*sin(yaw)*cos(pitch) + (-cv*z_calc/f + v1*z_calc/f)*(-sin(pitch)*sin(roll)*sin(yaw) - cos(roll)*cos(yaw)))/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))
    dv = f*(z_calc*(sin(pitch)*cos(roll)*cos(yaw) + sin(roll)*sin(yaw)) + (-cu*z_calc/f + u1*z_calc/f)*cos(pitch)*cos(yaw) + (-cv*z_calc/f + v1*z_calc/f)*(sin(pitch)*sin(roll)*cos(yaw) - sin(yaw)*cos(roll)))/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))

    return du, dv


def calculate_symbolic_derivatives():
    """
    Uses SymPy to calculate the symbolic form of the derivatives used above. 
    Substitutes a test point to check the derivative functions are all correct. 
    Does not accept any arguments or return anything, but prints all tests. 
    """

    # Setting up necessary symbolic variables
    z_calc, u1, v1, f, cu, cv, x, y, z, roll, pitch, yaw = sympy.symbols('z_calc, u1, v1, f, cu, cv, x, y, z, roll, pitch, yaw', real=True)

    # Calculating u2 and v2, the coordinates of the pixel in the second image
    K = Matrix([[f, 0, cu], [0, f, cv], [0, 0, 1]])

    T = zeros(4, 4)
    T[0:3, 3] = Matrix([[x], [y], [z]])
    T[3, 3] = 1

    cr = sympy.cos(roll)
    sr = sympy.sin(roll)
    cp = sympy.cos(pitch)
    sp = sympy.sin(pitch)
    cy = sympy.cos(yaw)
    sy = sympy.sin(yaw)

    R = Matrix([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                [  -sp,            cp*sr,            cp*cr]])

    T[0:3, 0:3] = R

    u_initial = Matrix([[u1], [v1], [1]])
    p = zeros(4, 1)
    p[0:3, 0] = z_calc * K.inv() * u_initial
    p[3] = 1
    p_trans = T * p
    u_trans = K * Matrix([[p_trans[0, 0]/p_trans[2, 0]], [p_trans[1, 0]/p_trans[2, 0]], [1]])

    u2 = u_trans[0, 0]
    v2 = u_trans[1, 0]

    # Set values for tests
    test_values = [(z_calc, 3.66), (u1, 163), (v1, 115), (f, 579.47), (cu, 374.77), (cv, 265.41), (x, 0), (y, -0.12), (z, 0), (roll, 0), (pitch, 0), (yaw, 0)]
    test_args = [val[1] for val in test_values]

    # Confirm u2 and v2 have been calculated correctly
    print(u2.subs(test_values), 163)
    print(v2.subs(test_values), 96)

    # Confirm all derivative functions are returning the same value as symbolic functions
    print(sympy.diff(u2, x).subs(test_values), get_dx(*test_args)[0])
    print(sympy.diff(v2, x).subs(test_values), get_dx(*test_args)[1])
    print(sympy.diff(u2, y).subs(test_values), get_dy(*test_args)[0])
    print(sympy.diff(v2, y).subs(test_values), get_dy(*test_args)[1])
    print(sympy.diff(u2, z).subs(test_values), get_dz(*test_args)[0])
    print(sympy.diff(v2, z).subs(test_values), get_dz(*test_args)[1])
    print(sympy.diff(u2, roll).subs(test_values), get_droll(*test_args)[0])
    print(sympy.diff(v2, roll).subs(test_values), get_droll(*test_args)[1])
    print(sympy.diff(u2, pitch).subs(test_values), get_dpitch(*test_args)[0])
    print(sympy.diff(v2, pitch).subs(test_values), get_dpitch(*test_args)[1])
    print(sympy.diff(u2, yaw).subs(test_values), get_dyaw(*test_args)[0])
    print(sympy.diff(v2, yaw).subs(test_values), get_dyaw(*test_args)[1])


if __name__ == "__main__":
    # Test all derivative calculations
    calculate_symbolic_derivatives()