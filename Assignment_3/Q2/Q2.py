import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def inverse_kinematics(x, y, z, l1, l2, l3):
    theta_1_1 = (-np.arccos(l1 / np.sqrt((x ** 2) + (y ** 2)))) + np.arctan2(-x, y)
    theta_1_2 = theta_1_1
    if np.isclose(np.sin(theta_1_1), 0):
        x_2r_1 = (x + (l1 * np.sin(theta_1_1))) / np.cos(theta_1_1)
    else:
        x_2r_1 = (y - (l1 * np.cos(theta_1_1))) / np.sin(theta_1_1)
    y_2r_1 = z
    a = 2 * l3 * x_2r_1
    b = 2 * l3 * y_2r_1
    c = (l2 ** 2) - (l3 ** 2) - (x_2r_1 ** 2) - (y_2r_1 ** 2)
    theta_23_1 = np.arccos(np.clip(-c / np.sqrt(a ** 2 + b ** 2),-1,1)) + np.arctan2(b, a)
    theta_2_1 = np.arctan2(y_2r_1 - (l3 * np.sin(theta_23_1)), x_2r_1 - (l3 * np.cos(theta_23_1)))    
    theta_3_1 = theta_23_1 - theta_2_1

    theta_23_2 = (-np.arccos(np.clip(-c / np.sqrt(a ** 2 + b ** 2),-1,1))) + np.arctan2(b, a)
    theta_2_2 = np.arctan2(y_2r_1 - (l3 * np.sin(theta_23_2)), x_2r_1 - (l3 * np.cos(theta_23_2)))
    theta_3_2 = theta_23_2 - theta_2_2

    
    theta_1_3 = (np.arccos(l1 / np.sqrt((x ** 2) + (y ** 2)))) + np.arctan2(-x, y)
    theta_1_4 = theta_1_3
    if np.isclose(np.sin(theta_1_3), 0):
        x_2r_2 = (x + (l1 * np.sin(theta_1_3))) / np.cos(theta_1_3)
    else:
        x_2r_2 = (y - (l1 * np.cos(theta_1_3))) / np.sin(theta_1_3)
    y_2r_2 = z
    a = 2 * l3 * x_2r_2
    b = 2 * l3 * y_2r_2
    c = (l2 ** 2) - (l3 ** 2) - (x_2r_2 ** 2) - (y_2r_2 ** 2)
    theta_23_3 = np.arccos(np.clip(-c / np.sqrt(a ** 2 + b ** 2),-1,1)) + np.arctan2(b, a)
    theta_2_3 = np.arctan2(y_2r_2 - (l3 * np.sin(theta_23_3)), x_2r_2 - (l3 * np.cos(theta_23_3)))    
    theta_3_3 = theta_23_3 - theta_2_3

    theta_23_4 = (-np.arccos(np.clip(-c / np.sqrt(a ** 2 + b ** 2),-1,1))) + np.arctan2(b, a)
    theta_2_4 = np.arctan2(y_2r_2 - (l3 * np.sin(theta_23_4)), x_2r_2 - (l3 * np.cos(theta_23_4)))
    theta_3_4 = theta_23_4 - theta_2_4

    return (np.mod(np.array([theta_1_1, theta_2_1, theta_3_1]),2*np.pi),
            np.mod(np.array([theta_1_2, theta_2_2, theta_3_2]),2*np.pi),
            np.mod(np.array([theta_1_3, theta_2_3, theta_3_3]),2*np.pi),
            np.mod(np.array([theta_1_4, theta_2_4, theta_3_4]),2*np.pi))

def dh_transform(alpha, a, d, theta):
    return np.array([
        [np.cos(theta),                -np.sin(theta),                  0,              a],
        [np.cos(alpha)*np.sin(theta),   np.cos(alpha)*np.cos(theta),   -np.sin(alpha), -d*np.sin(alpha)],
        [np.sin(alpha)*np.sin(theta),   np.cos(theta)*np.sin(alpha),    np.cos(alpha),  d*np.cos(alpha)],
        [0,                             0,                              0    ,          1]
    ])

def get_dh_matrices(link_lengths, joint_angles):
    return [
        dh_transform( 0,            0,                  0,                  joint_angles[0]),
        dh_transform( np.pi / 2,    0,                 -link_lengths[0],    joint_angles[1]),
        dh_transform( 0,            link_lengths[1],    0,                  joint_angles[2]),
        dh_transform(-np.pi / 2,    link_lengths[2],    0,                  0             )
    ]

def plot_manipulator_3d(ax, link_lengths, joint_angles):
    T = np.eye(4)
    x, y, z = [0], [0], [0]
    
    transformations = get_dh_matrices(link_lengths, joint_angles)
    for transform in transformations:
        T = T @ transform
        x.append(T[0, 3])
        y.append(T[1, 3])
        z.append(T[2, 3])
    
    ax.quiver(0, 0, 0, 0.4, 0, 0, linewidth=2, color='r')  # X-axis
    ax.quiver(0, 0, 0, 0, 0.4, 0, linewidth=2, color='g')  # Y-axis
    ax.quiver(0, 0, 0, 0, 0, 0.4, linewidth=2, color='b')  # Z-axis
    ax.plot(x, y, z, linewidth=2, color='grey')
    ax.scatter(x[-1], y[-1], z[-1], color='red')

    ax.view_init(elev=0, azim=180)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_box_aspect([1, 1, 1])
    ax.grid()

if __name__ == "__main__":
    link_lengths = [1.0,1.0,1.0]
    x, y, z = -1.5, -1.0, 1.0
    l1, l2, l3 = link_lengths
    configs = inverse_kinematics(x, y, z, l1, l2, l3)
    
    fig, axes = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(10, 8))
    axes[0, 0].set_title("Config 1")
    axes[0, 1].set_title("Config 2")
    axes[1, 0].set_title("Config 3")
    axes[1, 1].set_title("Config 4")

    plot_manipulator_3d(axes[0, 0], link_lengths, configs[0])
    plot_manipulator_3d(axes[0, 1], link_lengths, configs[1])
    plot_manipulator_3d(axes[1, 0], link_lengths, configs[2])
    plot_manipulator_3d(axes[1, 1], link_lengths, configs[3])

    plt.tight_layout()
    plt.show()