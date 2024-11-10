import numpy as np
from spatialmath.base import trotx, transl
from roboticstoolbox import DHRobot, RevoluteMDH

def dh_transform(alpha, a, d, theta):
    return np.array([
        [np.cos(theta),                -np.sin(theta),                  0,              a],
        [np.cos(alpha)*np.sin(theta),   np.cos(alpha)*np.cos(theta),   -np.sin(alpha), -d*np.sin(alpha)],
        [np.sin(alpha)*np.sin(theta),   np.cos(theta)*np.sin(alpha),    np.cos(alpha),  d*np.cos(alpha)],
        [0,                             0,                              0    ,          1]
    ])

# 1.2
def get_dh_matrices(link_lengths, joint_angles):
    transforms =  [
        dh_transform(0,             0,                  0,                  joint_angles[0]),
        dh_transform(np.pi / 2,     0,                 -link_lengths[0],    joint_angles[1]),
        dh_transform(0,             link_lengths[1],    0,                  joint_angles[2]),
        dh_transform(-np.pi / 2,     link_lengths[2],    0,                  0             )]
    return transforms

# 1.3
def forward_kinematics_through_dh(link_lengths, joint_angles):
    T = np.eye(4)
    transformations = get_dh_matrices(link_lengths, joint_angles)
    for transform in transformations:
        T = T @ transform
    return T

def forward_kinematics_through_eq(link_lengths, joint_angles):
    x = (((link_lengths[1]*np.cos(joint_angles[1])) + (link_lengths[2]*np.cos(joint_angles[1]+joint_angles[2])))*np.cos(joint_angles[0])) - (link_lengths[0]*np.sin(joint_angles[0]))
    y = (((link_lengths[1]*np.cos(joint_angles[1])) + (link_lengths[2]*np.cos(joint_angles[1]+joint_angles[2])))*np.sin(joint_angles[0])) + (link_lengths[0]*np.cos(joint_angles[0]))
    z = (link_lengths[1]*np.sin(joint_angles[1])) + (link_lengths[2]*np.sin(joint_angles[1]+joint_angles[2]))
    Rz = np.array([
        [np.cos(joint_angles[0]), -np.sin(joint_angles[0]), 0],
        [np.sin(joint_angles[0]),  np.cos(joint_angles[0]), 0],
        [0,                       0,                        1]
    ])

    Ry = np.array([
        [np.cos(-joint_angles[1] - joint_angles[2]),    0,      np.sin(-joint_angles[1] - joint_angles[2])   ],
        [0,                                             1,      0                                           ],
        [-np.sin(-joint_angles[1] - joint_angles[2]),   0,      np.cos(-joint_angles[1] - joint_angles[2])   ]
    ])

    R = Rz @ Ry

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

class OrthoParallel(DHRobot):
    def __init__(self):
        links = [
            RevoluteMDH(d=0, a=0, alpha=0),
            RevoluteMDH(d=-1, a=0, alpha=np.pi/2),
            RevoluteMDH(d=0, a=1, alpha=0),
            RevoluteMDH(d=0, a=1, alpha=-np.pi/2)
        ]
        super().__init__(
            links,
            name="OrthoParallel",
            manufacturer="Soham Patil",
        )

        self.qr = np.zeros(4)

        self.addconfiguration("qr", self.qr)

if __name__ == "__main__":
    link_lengths = [1.0, 1.0, 1.0]
    joint_angles = np.random.uniform(-np.pi, np.pi, size=3)
    print("Random joint angles:", joint_angles)

    # 1.2 Test
    transformation_matrices = get_dh_matrices(link_lengths, joint_angles)
    print("Transformation matrices for each joint:")
    for i, T in enumerate(transformation_matrices):
        print(f"Joint {i+1} to Joint {i}:\n{np.array_str(T, precision=4, suppress_small=True)}\n")

    # 1.3 Test
    pose_dh = forward_kinematics_through_dh(link_lengths, joint_angles)
    pose_eq = forward_kinematics_through_eq(link_lengths, joint_angles)

    print("End-Effector Pose using DH Parameters:\n", np.array_str(pose_dh, precision=4, suppress_small=True))
    print("End-Effector Pose using Equation Method:\n", np.array_str(pose_eq, precision=4, suppress_small=True))

    translation_match = np.allclose(pose_dh[:3, 3], pose_eq[:3, 3], atol=1e-4)
    rotation_match = np.allclose(pose_dh[:3, :3]@(pose_eq[:3, :3].T),np.eye(3), atol=1e-4)

    if translation_match and rotation_match:
        print("Test Passed: The end-effector poses match.")
    else:
        print("Test Failed: The end-effector poses do not match.")

    op = OrthoParallel()
    op.plot(q=op.qr,block=True)