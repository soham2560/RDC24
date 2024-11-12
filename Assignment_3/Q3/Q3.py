import sympy as sp
import numpy as np
from roboticstoolbox import DHRobot, RevoluteMDH

class OrthoParallel(DHRobot):
    def __init__(self):
        links = [
            RevoluteMDH(d=0, a=0, alpha=0),
            RevoluteMDH(d=-2, a=0, alpha=np.pi/2),
            RevoluteMDH(d=0, a=1.5, alpha=0),
            RevoluteMDH(d=0, a=1, alpha=-np.pi/2)
        ]
        super().__init__(
            links,
            name="OrthoParallel",
            manufacturer="Soham Patil",
        )

        self.qr = np.zeros(4)

        self.addconfiguration("qr", self.qr)

op = OrthoParallel()

theta1, theta2, theta3 = sp.symbols('t1 t2 t3')
dtheta = sp.symbols('dt1 dt2 dt3')
L1, L2, L3 = sp.symbols('L1 L2 L3')
z_axis = sp.Matrix([0, 0, 1])

def dh_transform(alpha, a, d, theta):
    return sp.Matrix([
        [sp.cos(theta),                -sp.sin(theta),                  0,              a                   ],
        [sp.sin(theta) * sp.cos(alpha), sp.cos(theta) * sp.cos(alpha), -sp.sin(alpha), -d * sp.sin(alpha)   ],
        [sp.sin(theta) * sp.sin(alpha), sp.cos(theta) * sp.sin(alpha),  sp.cos(alpha),  d * sp.cos(alpha)   ],
        [0,                             0,                              0,              1]
    ])

dh_params = [
    ( 0,         0,     0,  theta1  ),
    ( sp.pi / 2, 0,    -L1, theta2  ),
    ( 0,         L2,    0,  theta3  ),
    (-sp.pi / 2, L3,    0,  0       )
]

transforms = [dh_transform(*params) for params in dh_params]
rotations = [T[:3, :3] for T in transforms]
origins = [T[:3, 3] for T in transforms]

omega = [sp.Matrix([0, 0, 0])]
v = [sp.Matrix([0, 0, 0])]

for i in range(3):
    omega_j = rotations[i].T * omega[i] + (dtheta[i] * z_axis)
    omega.append(omega_j)
    v_j = rotations[i].T * (v[i] + omega[i].cross(origins[i]))
    v.append(v_j)

v_4 = rotations[3].T * (v[3] + omega[3].cross(origins[3]))
omega_4 = rotations[3].T * omega[3]

R04 = rotations[0] * rotations[1] * rotations[2] * rotations[3]

v_0 = R04 * v_4
omega_0 = R04 * omega_4

J_v = v_0.jacobian(dtheta)
J_omega = omega_0.jacobian(dtheta)
J = sp.Matrix.vstack(J_v, J_omega)

print("\nLinear velocities (v) for each frame:")
for i, v_i in enumerate(v):
    print(f"Frame {i}: {v_i}")

print("\nAngular velocities (omega) for each frame:")
for i, omega_i in enumerate(omega):
    print(f"Frame {i}: {omega_i}")
    
print("\nFull Jacobian for the manipulator (linear and angular velocities):")
sp.pprint(J.applyfunc(sp.simplify))

det_J_v = sp.simplify(J_v.det())

print("\nDeterminant of the linear velocity Jacobian matrix (J_v):")
sp.pprint(det_J_v)
solution_t3 = sp.solve(sp.sin(theta3), theta3)

solution_t2_t3 = sp.solve(L2 * sp.cos(theta2) + L3 * sp.cos(theta2 + theta3), theta3)

print("Solutions for sin(t3) = 0:")
for sol in solution_t3:
    sp.pprint(sp.Eq(theta3, sp.simplify(sol)))
    print(f"\nVisualising...")
    op.plot(q=[np.pi/4,np.pi/4,sol.evalf(),0], limits=[-2.5,2.5,-2.5,2.5,-2.5,2.5], dt = 5)

print("\nSolutions for L2 * cos(t2) + L3 * cos(t2 + t3) = 0:")
for sol in solution_t2_t3:
    sp.pprint(sp.Eq(theta3, sp.simplify(sol)))
    print(f"\nVisualising with theta2=2.0...")
    op.plot(q=[np.pi/4,np.pi/4,sol.subs({L2: op.links[2].a, L3: op.links[3].a,theta2:2.0}).evalf(),0], limits=[-2.5,2.5,-2.5,2.5,-2.5,2.5], dt = 5)