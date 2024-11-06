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
def forward_kinematics(link_lengths, joint_angles):
    T = np.eye(4)
    transformations = get_dh_matrices(link_lengths, joint_angles)
    for transform in transformations:
        T = T @ transform
    return T

class OrthoParallel(DHRobot):
    def __init__(self):
        links = [
            RevoluteMDH(d=0, a=0, alpha=0),
            RevoluteMDH(d=-1, a=0, alpha=np.pi/2),
            RevoluteMDH(d=0, a=1, alpha=0)
        ]
            
        tool = transl(1, 0, 0) @ trotx(-np.pi / 2) 
        super().__init__(
            links,
            name="OrthoParallel",
            manufacturer="Soham Patil",
            tool=tool
        )

        self.qr = np.zeros(3)

        self.addconfiguration("qr", self.qr)


if __name__ == "__main__":
    link_lengths = [1.0, 1.0, 1.0]
    joint_angles = [0.0, 0.0, 0.0]

    # 1.2 Test
    transformation_matrices = get_dh_matrices(link_lengths, joint_angles)
    print("Transformation matrices for each joint:")
    for i, T in enumerate(transformation_matrices):
        print(f"Joint {i+1} to Joint {i}:\n{np.array_str(T, precision=4, suppress_small=True)}\n")

    # 1.3 Test
    end_effector_pose = forward_kinematics(link_lengths, joint_angles)
    print("End-Effector Pose:\n", np.array_str(end_effector_pose, precision=4, suppress_small=True))

    op = OrthoParallel()
    op.plot(q=op.qr,block=True)