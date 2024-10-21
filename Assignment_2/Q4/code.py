import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def dh_transform(alpha, a, d, theta):
    return np.array([
        [np.cos(theta),                -np.sin(theta),                  0,              a],
        [np.cos(alpha)*np.sin(theta),   np.cos(alpha)*np.cos(theta),   -np.sin(alpha), -d*np.sin(alpha)],
        [np.sin(alpha)*np.sin(theta),   np.cos(theta)*np.sin(alpha),    np.cos(alpha),  d*np.cos(alpha)],
        [0,                             0,                              0    ,          1]
    ])

def forward_kinematics(link_lengths, joint_angles):
    transforms =  [
        dh_transform(0,             0,                  link_lengths[0],    joint_angles[0]             ),
        dh_transform(np.pi / 2,     0,                  0,                  joint_angles[1] + np.pi / 2 ),
        dh_transform(0,             link_lengths[1],    0,                  joint_angles[2] - np.pi / 2 ),
        dh_transform(0,             link_lengths[2],    link_lengths[3],    joint_angles[3]             ),
        dh_transform(-np.pi / 2,    0,                  link_lengths[4],    joint_angles[4] - np.pi / 2 ),
        dh_transform(-np.pi / 2,    0,                  link_lengths[5],    joint_angles[5]             )]
    T = np.eye(4)
    for transform in transforms:
        T = T @ transform
    return T, transforms

def plot_robot(link_lengths, joint_angles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([2, 2, 2]) 
    
    _, transformations = forward_kinematics(link_lengths, joint_angles)
    
    xs, ys, zs = [0], [0], [0]
    T_current = np.eye(4)
    
    for T_i in transformations:
        T_current = T_current @ T_i
        xs.append(T_current[0, 3])
        ys.append(T_current[1, 3])
        zs.append(T_current[2, 3])
    
    ax.plot(xs, ys, zs, '-o', label='Robot')
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('UR5 Robot Configuration')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([0, 3])
    
    plt.show()

link_lengths = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
home_angles = np.zeros(6)
random_angles = np.random.rand(6)

print("Testing Home Configuration:")
print(f"Joint Angles: {home_angles}")
T_home, _ = forward_kinematics(link_lengths, home_angles)
print(f"End-Effector Pose:\n{np.array_str(T_home, precision=4, suppress_small=True)}")

print("\nTesting Random Configuration:")
print(f"Joint Angles: {random_angles}")
T_random, _ = forward_kinematics(link_lengths, random_angles)
print(f"End-Effector Pose:\n{np.array_str(T_random, precision=4, suppress_small=True)}")

print("\nPlotting Home Configuration:")
plot_robot(link_lengths, home_angles)

print("\nPlotting Random Configuration:")
plot_robot(link_lengths, random_angles)
