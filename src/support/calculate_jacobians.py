from math import sin, cos
import sympy
from sympy.matrices import Matrix, zeros

def get_dx(z_calc, u1, v1, f, cu, cv, baseline, x, y, z, roll, pitch, yaw):
    du = f/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))
    dv = 0

    return du, dv


def get_dy(z_calc, u1, v1, f, cu, cv, baseline, x, y, z, roll, pitch, yaw):
    du = 0
    dv = f/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))

    return du, dv


def get_dz(z_calc, u1, v1, f, cu, cv, baseline, x, y, z, roll, pitch, yaw):
    du = -f*(x + z_calc*(sin(pitch)*cos(roll)*cos(yaw) + sin(roll)*sin(yaw)) + (-cu*z_calc/f + u1*z_calc/f)*cos(pitch)*cos(yaw) + (-cv*z_calc/f + v1*z_calc/f)*(sin(pitch)*sin(roll)*cos(yaw) - sin(yaw)*cos(roll)))/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))**2
    dv = -f*(y + z_calc*(sin(pitch)*sin(yaw)*cos(roll) - sin(roll)*cos(yaw)) + (-cu*z_calc/f + u1*z_calc/f)*sin(yaw)*cos(pitch) + (-cv*z_calc/f + v1*z_calc/f)*(sin(pitch)*sin(roll)*sin(yaw) + cos(roll)*cos(yaw)))/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))**2

    return du, dv
    

def get_droll(z_calc, u1, v1, f, cu, cv, baseline, x, y, z, roll, pitch, yaw):
    du = f*(z_calc*(-sin(pitch)*sin(roll)*cos(yaw) + sin(yaw)*cos(roll)) + (-cv*z_calc/f + v1*z_calc/f)*(sin(pitch)*cos(roll)*cos(yaw) + sin(roll)*sin(yaw)))/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch)) + f*(z_calc*sin(roll)*cos(pitch) - (-cv*z_calc/f + v1*z_calc/f)*cos(pitch)*cos(roll))*(x + z_calc*(sin(pitch)*cos(roll)*cos(yaw) + sin(roll)*sin(yaw)) + (-cu*z_calc/f + u1*z_calc/f)*cos(pitch)*cos(yaw) + (-cv*z_calc/f + v1*z_calc/f)*(sin(pitch)*sin(roll)*cos(yaw) - sin(yaw)*cos(roll)))/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))**2
    dv = f*(z_calc*(-sin(pitch)*sin(roll)*sin(yaw) - cos(roll)*cos(yaw)) + (-cv*z_calc/f + v1*z_calc/f)*(sin(pitch)*sin(yaw)*cos(roll) - sin(roll)*cos(yaw)))/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch)) + f*(z_calc*sin(roll)*cos(pitch) - (-cv*z_calc/f + v1*z_calc/f)*cos(pitch)*cos(roll))*(y + z_calc*(sin(pitch)*sin(yaw)*cos(roll) - sin(roll)*cos(yaw)) + (-cu*z_calc/f + u1*z_calc/f)*sin(yaw)*cos(pitch) + (-cv*z_calc/f + v1*z_calc/f)*(sin(pitch)*sin(roll)*sin(yaw) + cos(roll)*cos(yaw)))/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))**2

    return du, dv
    

def get_dpitch(z_calc, u1, v1, f, cu, cv, baseline, x, y, z, roll, pitch, yaw):
    du = f*(z_calc*sin(pitch)*cos(roll) - (cu*z_calc/f - u1*z_calc/f)*cos(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(pitch)*sin(roll))*(x + z_calc*(sin(pitch)*cos(roll)*cos(yaw) + sin(roll)*sin(yaw)) + (-cu*z_calc/f + u1*z_calc/f)*cos(pitch)*cos(yaw) + (-cv*z_calc/f + v1*z_calc/f)*(sin(pitch)*sin(roll)*cos(yaw) - sin(yaw)*cos(roll)))/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))**2 + f*(z_calc*cos(pitch)*cos(roll)*cos(yaw) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch)*cos(yaw) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch)*cos(yaw))/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))
    dv = f*(z_calc*sin(pitch)*cos(roll) - (cu*z_calc/f - u1*z_calc/f)*cos(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(pitch)*sin(roll))*(y + z_calc*(sin(pitch)*sin(yaw)*cos(roll) - sin(roll)*cos(yaw)) + (-cu*z_calc/f + u1*z_calc/f)*sin(yaw)*cos(pitch) + (-cv*z_calc/f + v1*z_calc/f)*(sin(pitch)*sin(roll)*sin(yaw) + cos(roll)*cos(yaw)))/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))**2 + f*(z_calc*sin(yaw)*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch)*sin(yaw) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*sin(yaw)*cos(pitch))/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))
    
    return du, dv
    

def get_dyaw(z_calc, u1, v1, f, cu, cv, baseline, x, y, z, roll, pitch, yaw):
    du = f*(z_calc*(-sin(pitch)*sin(yaw)*cos(roll) + sin(roll)*cos(yaw)) - (-cu*z_calc/f + u1*z_calc/f)*sin(yaw)*cos(pitch) + (-cv*z_calc/f + v1*z_calc/f)*(-sin(pitch)*sin(roll)*sin(yaw) - cos(roll)*cos(yaw)))/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))
    dv = f*(z_calc*(sin(pitch)*cos(roll)*cos(yaw) + sin(roll)*sin(yaw)) + (-cu*z_calc/f + u1*z_calc/f)*cos(pitch)*cos(yaw) + (-cv*z_calc/f + v1*z_calc/f)*(sin(pitch)*sin(roll)*cos(yaw) - sin(yaw)*cos(roll)))/(z + z_calc*cos(pitch)*cos(roll) - (-cu*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cv*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))

    return du, dv


def get_derivatives():
    z_calc, u1, v1, f, cu, cv, baseline, x, y, z, roll, pitch, yaw = sympy.symbols('z_calc, u1, v1, f, cu, cv, baseline, x, y, z, roll, pitch, yaw', real=True)

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

    test_values = [(z_calc, 3.66), (u1, 163), (v1, 115), (f, 579.47), (cu, 374.77), (cv, 265.41), (baseline, 0.12), (x, 0), (y, -0.12), (z, 0), (roll, 0), (pitch, 0), (yaw, 0)]
    test_args = [val[1] for val in test_values]

    print(u2.subs(test_values), 163)
    print(v2.subs(test_values), 96)

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
    get_derivatives()