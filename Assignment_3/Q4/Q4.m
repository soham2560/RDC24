syms theta1 theta2 theta1_dot theta2_dot real
syms m1 m2 l1 l2 r1 r2 g real

p1 = [r1 * cos(theta1); r1 * sin(theta1)];
p2 = [l1 * cos(theta1) + r2 * cos(theta1 + theta2); l1 * sin(theta1) + r2 * sin(theta1 + theta2)];

v1 = jacobian(p1, [theta1, theta2]) * [theta1_dot; theta2_dot];
v2 = jacobian(p2, [theta1, theta2]) * [theta1_dot; theta2_dot];

T = 0.5 * m1 * (v1' * v1) + 0.5 * m1 * r1^2 * theta1_dot^2 + ...
    0.5 * m2 * (v2' * v2) + 0.5 * m2 * r2^2 * (theta1_dot + theta2_dot)^2;

V = m1 * g * p1(2) + m2 * g * p2(2);

L = T - V;

M = [diff(T, theta1_dot, 2), diff(T, theta1_dot, theta2_dot);
     diff(T, theta2_dot, theta1_dot), diff(T, theta2_dot, 2)];

c11 = 0.5 * (diff(M(1,2), theta1) + diff(M(1,1), theta2) - diff(M(2,1), theta1)) * theta2_dot + ...
      0.5 * (diff(M(1,2), theta2) + diff(M(1,1), theta1) - diff(M(2,2), theta1)) * theta1_dot;

c12 = 0.5 * (diff(M(1,2), theta1) + diff(M(1,1), theta2) - diff(M(2,1), theta1)) * theta2_dot + ...
      0.5 * (diff(M(1,2), theta2) + diff(M(1,1), theta1) - diff(M(2,2), theta1)) * theta1_dot;

c21 = 0.5 * (diff(M(2,1), theta1) + diff(M(2,2), theta2) - diff(M(1,2), theta1)) * theta2_dot + ...
      0.5 * (diff(M(2,1), theta2) + diff(M(2,2), theta1) - diff(M(1,1), theta2)) * theta1_dot;

c22 = 0.5 * (diff(M(2,1), theta1) + diff(M(2,2), theta2) - diff(M(1,2), theta1)) * theta2_dot + ...
      0.5 * (diff(M(2,1), theta2) + diff(M(2,2), theta1) - diff(M(1,1), theta2)) * theta1_dot;

C = [c11, c12; c21, c22];

G = [-diff(V, theta1); -diff(V, theta2)];

disp('Mass Matrix M:');
disp(simplify(M));

disp('Coriolis Matrix C:');
disp(simplify(C));

disp('Gravitational Forces G:');
disp(simplify(G));

theta_ddot = M \ (-C * [theta1_dot; theta2_dot] - G);
theta1_ddot_func = matlabFunction(theta_ddot(1), 'Vars', {theta1, theta2, theta1_dot, theta2_dot, m1, m2, r1, r2, l1, l2, g});
theta2_ddot_func = matlabFunction(theta_ddot(2), 'Vars', {theta1, theta2, theta1_dot, theta2_dot, m1, m2, r1, r2, l1, l2, g});

params = struct('m1', 1, 'm2', 1, 'r1', 0.5, 'r2', 0.5, 'l1', 1, 'l2', 1, 'g', -9.81, ...
                'theta1_ddot_func', theta1_ddot_func, 'theta2_ddot_func', theta2_ddot_func);

initialAngles = [pi/4; -pi/4];
initialVelocities = [0; 0];

function dState = manipulatorDynamics(~, state, params)
    theta1 = state(1); theta2 = state(2);
    theta1_dot = state(3); theta2_dot = state(4);

    theta1_ddot = params.theta1_ddot_func(theta1, theta2, theta1_dot, theta2_dot, ...
                                          params.m1, params.m2, params.r1, params.r2, params.l1, params.l2, params.g);
    theta2_ddot = params.theta2_ddot_func(theta1, theta2, theta1_dot, theta2_dot, ...
                                          params.m1, params.m2, params.r1, params.r2, params.l1, params.l2, params.g);
    dState = [theta1_dot; theta2_dot; theta1_ddot; theta2_ddot];
end

tspan = [0 10];
initialState = [initialAngles; initialVelocities];
options = odeset('RelTol', 1e-6, 'AbsTol', 1e-6);
[t, states] = ode45(@(t, state) manipulatorDynamics(t, state, params), tspan, initialState, options);

figure;
plot(t, states(:,1), 'r', 'DisplayName', '\theta_1(t)');
hold on;
plot(t, states(:,2), 'b', 'DisplayName', '\theta_2(t)');
legend;
xlabel('Time (s)');
ylabel('Joint Angles (rad)');
title('Joint Angles Over Time');
saveas(gcf, 'JointAngles.png');

figure;
axis equal;
xlim([-2 2]);
ylim([-2 2]);
title('Manipulator Animation');

videoWriter = VideoWriter('ManipulatorMotion.avi');
videoWriter.FrameRate = 20;
open(videoWriter);

for k = 1:length(t)
    joint1 = [params.l1 * cos(states(k,1)), params.l1 * sin(states(k,1))];
    endEffector = joint1 + [params.l2 * cos(states(k,1) + states(k,2)), ...
                            params.l2 * sin(states(k,1) + states(k,2))];
    
    cla;
    plot([0, joint1(1), endEffector(1)], [0, joint1(2), endEffector(2)], '-o', 'LineWidth', 2);
    axis equal;
    xlim([-2 2]); ylim([-2 2]);
    drawnow;
    
    frame = getframe(gcf);
    writeVideo(videoWriter, frame);
end

close(videoWriter);

disp('Joint angles image saved as "JointAngles.png".');
disp('Manipulator motion video saved as "ManipulatorMotion.avi".');
