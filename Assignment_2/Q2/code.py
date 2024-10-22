import numpy as np
import matplotlib.pyplot as plt

def forward_kinematics(theta1, theta2, theta3, l1, l2, l3):
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2) + l3 * np.cos(theta1 + theta2 + theta3)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2) + l3 * np.sin(theta1 + theta2 + theta3)
    phi = theta1 + theta2 + theta3 
    return x, y, phi

def plot_manipulator(ax, theta1, theta2, theta3, l1, l2, l3):
    x0, y0 = 0, 0
    x1, y1 = l1 * np.cos(theta1), l1 * np.sin(theta1)
    x2, y2 = x1 + l2 * np.cos(theta1 + theta2), y1 + l2 * np.sin(theta1 + theta2)
    x3, y3 = x2 + l3 * np.cos(theta1 + theta2 + theta3), y2 + l3 * np.sin(theta1 + theta2 + theta3)

    ax.plot([x0, x1, x2, x3], [y0, y1, y2, y3], 'ro-', linewidth=2)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.grid()
    ax.set_aspect('equal', adjustable='box')

def plot_workspace(ax, l1, l2, l3):
    ax.add_patch(plt.Circle((0, 0), l1 + l2 - l3, color='lightblue', alpha=0.5))
    ax.add_patch(plt.Circle((0, 0), l1 - l2 + l3, color='white'))

def inverse_kinematics(x, y, phi, l1, l2, l3):
    x_2r = x - (l3 * np.cos(phi))
    y_2r = y - (l3 * np.sin(phi))
    a = 2 * l2 * x_2r
    b = 2 * l2 * y_2r
    c = (l1 ** 2) - (l2 ** 2) - (x_2r ** 2) - (y_2r ** 2)

    theta_12_1 = np.arccos(-c / np.sqrt(a ** 2 + b ** 2)) + np.arctan2(b, a)
    theta_1_1 = np.arctan2(y_2r - (l2 * np.sin(theta_12_1)), x_2r - (l2 * np.cos(theta_12_1)))    
    theta_2_1 = theta_12_1 - theta_1_1
    theta_3_1 = phi - theta_12_1

    theta_12_2 = (-np.arccos(-c / np.sqrt(a ** 2 + b ** 2))) + np.arctan2(b, a)
    theta_1_2 = np.arctan2(y_2r - (l2 * np.sin(theta_12_2)), x_2r - (l2 * np.cos(theta_12_2)))
    theta_2_2 = theta_12_2 - theta_1_2
    theta_3_2 = phi - theta_12_2

    return (np.mod(np.array([theta_1_1, theta_2_1, theta_3_1]),2*np.pi),
            np.mod(np.array([theta_1_2, theta_2_2, theta_3_2]),2*np.pi))


l1, l2, l3 = 2.0, 1.5, 1.0
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for ax in axs.flatten():
    theta1, theta2, theta3 = np.random.uniform(0, 2 * np.pi, 3)
    x, y, phi = forward_kinematics(theta1, theta2, theta3, l1, l2, l3)
    plot_workspace(ax, l1, l2, l3)
    plot_manipulator(ax, theta1, theta2, theta3, l1, l2, l3)
    ax.set_title(f"Manipulator Configuration (θ1={theta1:.2f}, θ2={theta2:.2f}, θ3={theta3:.2f})")
    
plt.tight_layout()
plt.show()

# Compare FK IK
angles = np.random.uniform(0, 2 * np.pi, 3)
x_target, y_target, phi_target = forward_kinematics(angles[0], angles[1], angles[2], l1, l2, l3)
elbow_down_angles, elbow_up_angles = inverse_kinematics(x_target, y_target, phi_target, l1, l2, l3)

# Print comparison of angles between FK and IK
print("Comparing Forward and Inverse Kinematics results:")
print("\nInput joint angles from FK (θ1, θ2, θ3):")
print(np.array_str(angles, precision=4, suppress_small=True))
print("\nRecovered angles from IK (Elbow-down solution):")
print(np.array_str(elbow_down_angles, precision=4, suppress_small=True))
print("\nRecovered angles from IK (Elbow-up solution):")
print(np.array_str(elbow_up_angles, precision=4, suppress_small=True))

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plot_manipulator(plt.gca(), angles[0], angles[1], angles[2], l1, l2, l3)
plt.title("Forward Kinematics")

plt.subplot(1, 3, 2)
plot_manipulator(plt.gca(), elbow_down_angles[0], elbow_down_angles[1], elbow_down_angles[2], l1, l2, l3)
plt.title("Inverse Kinematics (Elbow-down)")

plt.subplot(1, 3, 3)
plot_manipulator(plt.gca(), elbow_up_angles[0], elbow_up_angles[1], elbow_up_angles[2], l1, l2, l3)
plt.title("Inverse Kinematics (Elbow-up)")

plt.show()
