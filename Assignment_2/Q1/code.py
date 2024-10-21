import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rm2e(rm):
    if np.isclose(rm[2][0], -1):
        az = 0
        ay = np.pi / 2
        ax = -np.arctan2(rm[1][2], rm[0][2])
    elif np.isclose(rm[2][0], 1):
        az = 0
        ay = -np.pi / 2
        ax = np.arctan2(-rm[1][2], -rm[0][2])
    else:
        az = np.arctan2(rm[1][0], rm[0][0])
        ay = np.arctan2(-rm[2][0], np.sqrt(rm[0][0]**2 + rm[1][0]**2))
        ax = np.arctan2(rm[2][1], rm[2][2])
    return np.array([az, ay, ax])

def e2rm(angles):
    theta1 = angles[0]
    theta2 = angles[1]
    theta3 = angles[2]
    R_z = np.array([
        [np.cos(theta1), -np.sin(theta1), 0],
        [np.sin(theta1), np.cos(theta1), 0],
        [0, 0, 1]
    ])
    
    R_y = np.array([
        [np.cos(theta2), 0, np.sin(theta2)],
        [0, 1, 0],
        [-np.sin(theta2), 0, np.cos(theta2)]
    ])
    
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta3), -np.sin(theta3)],
        [0, np.sin(theta3), np.cos(theta3)]
    ])
    R = R_z @ R_y @ R_x
    return R

def plot_frames(angles):
    R = e2rm(angles)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Origin and basis of the {S} frame
    origin = np.array([0, 0, 0])
    S_basis = np.eye(3)
    B_basis = R @ S_basis

    # Plot the {S} frame
    ax.quiver(*origin, *S_basis[:, 0], color='r', label='{S} x-axis')
    ax.quiver(*origin, *S_basis[:, 1], color='g', label='{S} y-axis')
    ax.quiver(*origin, *S_basis[:, 2], color='b', label='{S} z-axis')

    # Plot the {B} frame
    ax.quiver(*origin, *B_basis[:, 0], color='r', linestyle='--', label='{B} x-axis')
    ax.quiver(*origin, *B_basis[:, 1], color='g', linestyle='--', label='{B} y-axis')
    ax.quiver(*origin, *B_basis[:, 2], color='b', linestyle='--', label='{B} z-axis')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

euler_angles = np.random.rand(3)
print("Let's test Non-Singular Configuration:")
print("\nRandomly generated angles are:")
print(np.array_str(euler_angles, precision=4, suppress_small=True))
print("\nEuler Angle to Rotation Matrix Conversion gives:")
e2rm_matrix = e2rm(euler_angles)
print(np.array_str(e2rm_matrix, precision=4, suppress_small=True))
print("\nRecovered Angles:")
recovered_euler_angles = rm2e(e2rm_matrix)
print(np.array_str(recovered_euler_angles, precision=4, suppress_small=True))
print("\nComparison for Non-Singular Config:")
print(f"{np.array2string(euler_angles, precision=4, suppress_small=True)}->{np.array2string(recovered_euler_angles, precision=4, suppress_small=True)}")

print("\nLet's test Singular Configuration by setting Theta_2 = pi/2:")
euler_angles[1] = np.pi/2
print("\nSingular Angles are:")
print("\nEuler Angle to Rotation Matrix Conversion gives:")
e2rm_matrix = e2rm(euler_angles)
print(np.array_str(e2rm_matrix, precision=4, suppress_small=True))
print("\nSingle Set of Recovered Angles by setting Theta_1 to 0.0:")
recovered_euler_angles = rm2e(e2rm_matrix)
print(np.array_str(recovered_euler_angles, precision=4, suppress_small=True))
print("\nComparison for Singular Config:")
print(f"{np.array2string(euler_angles, precision=4, suppress_small=True)}->{np.array2string(recovered_euler_angles, precision=4, suppress_small=True)}")

print("\nLet's plot rotation for random angles:")
euler_angles = np.random.rand(3)
print("\nRandomly generated angles are:")
print(np.array_str(euler_angles, precision=4, suppress_small=True))
plot_frames(euler_angles)