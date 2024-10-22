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
    [r11, r12, r13] = matrix[0, :]
    [r21, r22, r23] = matrix[1, :]
    [r31, r32, r33] = matrix[2, :]

    vector = [r11+r22+r33,
              r11,
              r22,
              r33]
    
    max_id = vector.index(max(vector))

    if max_id == 0:
        return 0.5 * np.array([
            np.sqrt(1 + r11 + r22 + r33),
            (r32 - r23) / np.sqrt(1 + r11 + r22 + r33),
            (r13 - r31) / np.sqrt(1 + r11 + r22 + r33),
            (r21 - r12) / np.sqrt(1 + r11 + r22 + r33)
        ])
    elif max_id == 1:
        return 0.5 * np.array([
            (r32 - r23) / np.sqrt(1 + r11 - r22 - r33),
            np.sqrt(1 + r11 - r22 - r33),
            (r12 + r21) / np.sqrt(1 + r11 - r22 - r33),
            (r31 + r13) / np.sqrt(1 + r11 - r22 - r33)
        ])
    elif max_id == 2:
        return 0.5 * np.array([
            (r13 - r31) / np.sqrt(1 - r11 + r22 - r33),
            (r12 + r21) / np.sqrt(1 - r11 + r22 - r33),
            np.sqrt(1 - r11 + r22 - r33),
            (r23 + r32) / np.sqrt(1 - r11 + r22 - r33)
        ])
    elif max_id == 3:
        return 0.5 * np.array([
            (r21 - r12) / np.sqrt(1 - r11 - r22 + r33),
            (r31 + r13) / np.sqrt(1 - r11 - r22 + r33),
            (r23 + r32) / np.sqrt(1 - r11 - r22 + r33),
            np.sqrt(1 - r11 - r22 + r33)
        ])

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def quaternion_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def rotate_vector_by_quaternion(v, q):
    v_quat = np.array([0, *v])
    q_conjugate = quaternion_conjugate(q)
    rotated_v_quat = quaternion_multiply(quaternion_multiply(q, v_quat), q_conjugate)
    return rotated_v_quat[1:]

def plot_rotation_matrix_and_axis(R, axis, theta):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    origin = np.array([0, 0, 0])
    S_basis = np.eye(3)

    ax.quiver(*origin, *S_basis[:, 0], color='r', label='{S} x-axis')
    ax.quiver(*origin, *S_basis[:, 1], color='g', label='{S} y-axis')
    ax.quiver(*origin, *S_basis[:, 2], color='b', label='{S} z-axis')

    B_basis = R @ S_basis
    ax.quiver(*origin, *B_basis[:, 0], color='r', linestyle='--', label='{B} x-axis')
    ax.quiver(*origin, *B_basis[:, 1], color='g', linestyle='--', label='{B} y-axis')
    ax.quiver(*origin, *B_basis[:, 2], color='b', linestyle='--', label='{B} z-axis')

    ax.quiver(*origin, *axis, color='m', linewidth=2, label='Rotation Axis')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.title(f'Rotation Matrix and Axis (Theta = {np.rad2deg(theta):.2f}°)')
    plt.show()

def plot_frames_with_quaternion(q):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    origin = np.array([0, 0, 0])
    S_basis = np.eye(3)

    ax.quiver(*origin, *S_basis[:, 0], color='r', label='{S} x-axis')
    ax.quiver(*origin, *S_basis[:, 1], color='g', label='{S} y-axis')
    ax.quiver(*origin, *S_basis[:, 2], color='b', label='{S} z-axis')

    B_basis_x = rotate_vector_by_quaternion(S_basis[:, 0], q)
    B_basis_y = rotate_vector_by_quaternion(S_basis[:, 1], q)
    B_basis_z = rotate_vector_by_quaternion(S_basis[:, 2], q)

    ax.quiver(*origin, *B_basis_x, color='r', linestyle='--', label='{B} x-axis')
    ax.quiver(*origin, *B_basis_y, color='g', linestyle='--', label='{B} y-axis')
    ax.quiver(*origin, *B_basis_z, color='b', linestyle='--', label='{B} z-axis')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.title('Original Frame {S} and Transformed Frame {B} using Quaternion Rotation')
    plt.show()

n = np.random.rand(3) - 0.5
theta = np.random.rand() * 2 * np.pi
n /= np.linalg.norm(n)

R_from_axis_angle = axisangle2rm(theta, n)

axis_recovered, theta_recovered = rm2axisangle(R_from_axis_angle)

print("Let's test the Axis-Angle to Rotation Matrix Conversion:")
print("\nRandomly generated axis and angle are:")
print("Axis:", np.array_str(n, precision=4, suppress_small=True))
print(f"Angle (radians): {theta:.4f}")

print("\nAxis-Angle to Rotation Matrix Conversion gives:")
print(np.array_str(R_from_axis_angle, precision=4, suppress_small=True))

print("\nRecovered Axis and Angle from Rotation Matrix:")
print("Axis:", np.array_str(axis_recovered, precision=4, suppress_small=True))
print(f"Angle (radians): {theta_recovered:.4f}")

print("\nComparison for Axis-Angle Representation:")
print(f"Axis: {np.array2string(n, precision=4, suppress_small=True)} -> {np.array2string(axis_recovered, precision=4, suppress_small=True)}")
print(f"Angle: {theta:.4f} -> {theta_recovered:.4f}")

q = rm2q(R_from_axis_angle)
print("\nQuaternion Representation from Rotation Matrix:")
print(np.array_str(q, precision=4, suppress_small=True))

print("\nLet's plot the rotation for the generated axis and angle:")
plot_rotation_matrix_and_axis(R_from_axis_angle, n, theta)

print("\nLet's visualize the rotation using the quaternion:")
plot_frames_with_quaternion(q)

