import numpy as np
from roboticstoolbox import DHRobot, RevoluteMDH

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

    return np.array([
        np.mod(np.array([theta_1_1, theta_2_1, theta_3_1, 0.0]),2*np.pi),
        np.mod(np.array([theta_1_2, theta_2_2, theta_3_2, 0.0]),2*np.pi),
        np.mod(np.array([theta_1_3, theta_2_3, theta_3_3, 0.0]),2*np.pi),
        np.mod(np.array([theta_1_4, theta_2_4, theta_3_4, 0.0]),2*np.pi)])

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
    x, y, z = -1.5, -1.0, 1.0
    l1, l2, l3 = link_lengths
    configs = inverse_kinematics(x, y, z, l1, l2, l3)

    op = OrthoParallel()
    op.plot(q=configs, dt = 1, limits=[-1.5,1.5,-1.5,1.5,-1.5,1.5], loop=True)