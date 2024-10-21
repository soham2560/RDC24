import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rm2axisangle(R):
    if np.isclose(R, np.eye(3)).all():
        return np.array([np.nan, np.nan, np.nan]), 0.0
    elif np.isclose(np.trace(R), -1):
        denom = 1 / np.sqrt(2 * (1 + R[2, 2])) if R[2, 2] != -1 else np.nan
        return denom * np.array([R[0, 2], R[1, 2], 1 + R[2, 2]]), np.pi

    theta = np.arccos((np.trace(R) - 1) / 2)
    denom = 1 / (2 * np.sin(theta)) if np.sin(theta) != 0 else np.nan
    n1 = (R[2, 1] - R[1, 2])
    n2 = (R[0, 2] - R[2, 0])
    n3 = (R[1, 0] - R[0, 1])  
    return denom * np.array([n1, n2, n3]), theta

def axisangle2rm(theta, axis):
    axis = axis / np.linalg.norm(axis)
    outer_product = np.outer(axis, axis)
    skew_axis = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return (1 - np.cos(theta)) * outer_product + np.cos(theta) * np.eye(3) + np.sin(theta) * skew_axis

def rm2q(matrix):
    r11, r12, r13 = matrix[0, :]
    r21, r22, r23 = matrix[1, :]
    r31, r32, r33 = matrix[2, :]

    vector = [r11 + r22 + r33, r11, r22, r33]
    max_id = vector.index(max(vector))

    if max_id == 0:
        s = 2 * np.sqrt(1 + r11 + r22 + r33)
        q = 0.5 * np.array([
            s,
            (r32 - r23) / s,
            (r13 - r31) / s,
            (r21 - r12) / s
        ])
    elif max_id == 1:
        s = 2 * np.sqrt(1 + r11 - r22 - r33)
        q = 0.5 * np.array([
            (r32 - r23) / s,
            s,
            (r12 + r21) / s,
            (r31 + r13) / s
        ])
    elif max_id == 2:
        s = 2 * np.sqrt(1 - r11 + r22 - r33)
        q = 0.5 * np.array([
            (r13 - r31) / s,
            (r12 + r21) / s,
            s,
            (r23 + r32) / s
        ])
    elif max_id == 3:
        s = 2 * np.sqrt(1 - r11 - r22 + r33)
        q = 0.5 * np.array([
            (r21 - r12) / s,
            (r31 + r13) / s,
            (r23 + r32) / s,
            s
        ])
    else:
        return np.array([0, 0, 0, 1])

    return q / np.linalg.norm(q)

def e2rm(angles):
    theta1, theta2, theta3 = angles
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
    return R_z @ R_y @ R_x

def plot_rotation_matrix(R):
    
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
    plt.show()

def plot_quaternion(q):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    origin = np.zeros(3)
    ax.quiver(*origin, q[1], q[2], q[3], color='m', length=1, normalize=True)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_title('Quaternion Visualization')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

euler_angles = np.random.rand(3) * 2 * np.pi
R = e2rm(euler_angles)

n, theta = rm2axisangle(R)
R_recovered = axisangle2rm(theta, n)
q = rm2q(R)

print("Rotation Matrix:\n", np.array_str(R, precision=4, suppress_small=True))
print("\nAxis:\n", np.array_str(n, precision=4, suppress_small=True))
print("\nTheta:\n", theta)
print("\nQuaternion:\n", np.array_str(q, precision=4, suppress_small=True))

plot_rotation_matrix(R)
plot_quaternion(q)