from math import sin, cos

def get_dx(z_calc, u1, v1, f, cx, cy, baseline, x, y, z, roll, pitch, yaw):
    du = 0
    dv = f/(z + z_calc*cos(pitch)*cos(roll) - (-cx*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cy*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))

    return du, dv


def get_dy(z_calc, u1, v1, f, cx, cy, baseline, x, y, z, roll, pitch, yaw):
    du = f/(z + z_calc*cos(pitch)*cos(roll) - (-cx*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cy*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))
    dv = 0

    return du, dv


def get_dz(z_calc, u1, v1, f, cx, cy, baseline, x, y, z, roll, pitch, yaw):
    du = -f*(y + z_calc*(sin(pitch)*sin(yaw)*cos(roll) - sin(roll)*cos(yaw)) + (-cx*z_calc/f + u1*z_calc/f)*sin(yaw)*cos(pitch) + (-cy*z_calc/f + v1*z_calc/f)*(sin(pitch)*sin(roll)*sin(yaw) + cos(roll)*cos(yaw)))/(z + z_calc*cos(pitch)*cos(roll) - (-cx*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cy*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))**2
    dv = -f*(x + z_calc*(sin(pitch)*cos(roll)*cos(yaw) + sin(roll)*sin(yaw)) + (-cx*z_calc/f + u1*z_calc/f)*cos(pitch)*cos(yaw) + (-cy*z_calc/f + v1*z_calc/f)*(sin(pitch)*sin(roll)*cos(yaw) - sin(yaw)*cos(roll)))/(z + z_calc*cos(pitch)*cos(roll) - (-cx*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cy*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))**2

    return du, dv
    

def get_droll(z_calc, u1, v1, f, cx, cy, baseline, x, y, z, roll, pitch, yaw):
    du = f*(z_calc*(-sin(pitch)*sin(roll)*sin(yaw) - cos(roll)*cos(yaw)) + (-cy*z_calc/f + v1*z_calc/f)*(sin(pitch)*sin(yaw)*cos(roll) - sin(roll)*cos(yaw)))/(z + z_calc*cos(pitch)*cos(roll) - (-cx*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cy*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch)) + f*(z_calc*sin(roll)*cos(pitch) - (-cy*z_calc/f + v1*z_calc/f)*cos(pitch)*cos(roll))*(y + z_calc*(sin(pitch)*sin(yaw)*cos(roll) - sin(roll)*cos(yaw)) + (-cx*z_calc/f + u1*z_calc/f)*sin(yaw)*cos(pitch) + (-cy*z_calc/f + v1*z_calc/f)*(sin(pitch)*sin(roll)*sin(yaw) + cos(roll)*cos(yaw)))/(z + z_calc*cos(pitch)*cos(roll) - (-cx*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cy*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))**2
    dv = f*(z_calc*(-sin(pitch)*sin(roll)*cos(yaw) + sin(yaw)*cos(roll)) + (-cy*z_calc/f + v1*z_calc/f)*(sin(pitch)*cos(roll)*cos(yaw) + sin(roll)*sin(yaw)))/(z + z_calc*cos(pitch)*cos(roll) - (-cx*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cy*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch)) + f*(z_calc*sin(roll)*cos(pitch) - (-cy*z_calc/f + v1*z_calc/f)*cos(pitch)*cos(roll))*(x + z_calc*(sin(pitch)*cos(roll)*cos(yaw) + sin(roll)*sin(yaw)) + (-cx*z_calc/f + u1*z_calc/f)*cos(pitch)*cos(yaw) + (-cy*z_calc/f + v1*z_calc/f)*(sin(pitch)*sin(roll)*cos(yaw) - sin(yaw)*cos(roll)))/(z + z_calc*cos(pitch)*cos(roll) - (-cx*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cy*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))**2

    return du, dv
    

def get_dpitch(z_calc, u1, v1, f, cx, cy, baseline, x, y, z, roll, pitch, yaw):
    du = f*(z_calc*sin(pitch)*cos(roll) - (cx*z_calc/f - u1*z_calc/f)*cos(pitch) + (-cy*z_calc/f + v1*z_calc/f)*sin(pitch)*sin(roll))*(y + z_calc*(sin(pitch)*sin(yaw)*cos(roll) - sin(roll)*cos(yaw)) + (-cx*z_calc/f + u1*z_calc/f)*sin(yaw)*cos(pitch) + (-cy*z_calc/f + v1*z_calc/f)*(sin(pitch)*sin(roll)*sin(yaw) + cos(roll)*cos(yaw)))/(z + z_calc*cos(pitch)*cos(roll) - (-cx*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cy*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))**2 + f*(z_calc*sin(yaw)*cos(pitch)*cos(roll) - (-cx*z_calc/f + u1*z_calc/f)*sin(pitch)*sin(yaw) + (-cy*z_calc/f + v1*z_calc/f)*sin(roll)*sin(yaw)*cos(pitch))/(z + z_calc*cos(pitch)*cos(roll) - (-cx*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cy*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))
    dv = f*(z_calc*sin(pitch)*cos(roll) - (cx*z_calc/f - u1*z_calc/f)*cos(pitch) + (-cy*z_calc/f + v1*z_calc/f)*sin(pitch)*sin(roll))*(x + z_calc*(sin(pitch)*cos(roll)*cos(yaw) + sin(roll)*sin(yaw)) + (-cx*z_calc/f + u1*z_calc/f)*cos(pitch)*cos(yaw) + (-cy*z_calc/f + v1*z_calc/f)*(sin(pitch)*sin(roll)*cos(yaw) - sin(yaw)*cos(roll)))/(z + z_calc*cos(pitch)*cos(roll) - (-cx*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cy*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))**2 + f*(z_calc*cos(pitch)*cos(roll)*cos(yaw) - (-cx*z_calc/f + u1*z_calc/f)*sin(pitch)*cos(yaw) + (-cy*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch)*cos(yaw))/(z + z_calc*cos(pitch)*cos(roll) - (-cx*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cy*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))
    
    return du, dv
    

def get_dyaw(z_calc, u1, v1, f, cx, cy, baseline, x, y, z, roll, pitch, yaw):
    du = f*(z_calc*(sin(pitch)*cos(roll)*cos(yaw) + sin(roll)*sin(yaw)) + (-cx*z_calc/f + u1*z_calc/f)*cos(pitch)*cos(yaw) + (-cy*z_calc/f + v1*z_calc/f)*(sin(pitch)*sin(roll)*cos(yaw) - sin(yaw)*cos(roll)))/(z + z_calc*cos(pitch)*cos(roll) - (-cx*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cy*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))
    dv = f*(z_calc*(-sin(pitch)*sin(yaw)*cos(roll) + sin(roll)*cos(yaw)) - (-cx*z_calc/f + u1*z_calc/f)*sin(yaw)*cos(pitch) + (-cy*z_calc/f + v1*z_calc/f)*(-sin(pitch)*sin(roll)*sin(yaw) - cos(roll)*cos(yaw)))/(z + z_calc*cos(pitch)*cos(roll) - (-cx*z_calc/f + u1*z_calc/f)*sin(pitch) + (-cy*z_calc/f + v1*z_calc/f)*sin(roll)*cos(pitch))

    return du, dv
