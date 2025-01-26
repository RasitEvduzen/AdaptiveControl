% Direct MRAC for Second-Order SISO System
% Written By: Rasit
% Date: 26-Jan-2025
clc,clear,close all;
%% System Parameters (True and Reference Model)
w = 1; zeta = -0.5; b = 1;
A = [0 1; -w^2 -2*zeta*w]; 
B = [0; b]; 
theta = [0.5; -0.1];
phi = @(x) [1; x(1)^2]; % Nonlinear basis function

% Reference Model
Am = [0 1; -4 -2]; 
bm = 4; 
Bm = [0; bm];

% Solve Lyapunov Equation
Q = eye(2);
P = lyap(Am', Q);
Pbar = [P(1,2); P(2,2)];

%% Simulation Parameters
Ts = 1e-2; 
t = 0:Ts:50; % Time step and simulation duration
ref = zeros(length(t),1);      % Input signal (initially zero)
% ref(t > 0.5) = sin(t(t > 0.5)); % Sinusoidal input after 0.5 second
ref(t > 1) = square(.5*t(t > 1));    % Square input after 1 second

% Initial Conditions
x_ref = zeros(2, length(t));
x = zeros(2, length(t));
x_no_adapt = zeros(2, length(t));
u = zeros(length(t), 1);
u_no_adapt = zeros(length(t), 1);
kx = zeros(length(t), 2);
kr = zeros(length(t), 1);
theta_est = zeros(2, length(t));


%% Adaptive Control Gains
gamma_x = diag([100 100]);
gamma_r = 100;
gamma_theta = diag([100 100]);

% True Control Gains (for non-adaptive comparison)
kx_true = pinv(B'*B) * B' * (Am - A); % Pseudo-inverse for B
kr_true = bm / b;

%% Simulation Loop
figure('units', 'normalized', 'outerposition', [0 0 1 1], 'color', 'w');
for i = 2:length(t)
    if (i*Ts) > 25
        ref(i) = cos(i*Ts);
    end

    % Control Law
    u(i-1) = kx_true * x(:,i-1) + kr_true * ref(i) - theta_est(:,i-1)' * phi(x(:,i-1));
    u_no_adapt(i-1) = kx_true * x_no_adapt(:,i-1) + kr_true * ref(i);

    % Adaptive Parameter Update
    e = x_ref(:,i-1) - x(:,i-1);
    kx_dot = gamma_x * x(:,i-1) * e' * Pbar * sign(b);
    kr_dot = gamma_r * ref(i) * e' * Pbar * sign(b);
    theta_dot = -gamma_theta * phi(x(:,i-1)) * e' * Pbar * sign(b);

    % System Dynamics
    xdot_ref = Am * x_ref(:,i-1) + Bm * ref(i);
    xdot = A * x(:,i-1) + B * u(i-1) + B * theta' * phi(x(:,i-1));
    xdot_no_adapt = A * x_no_adapt(:,i-1) + B * u_no_adapt(i-1) + B * theta' * phi(x_no_adapt(:,i-1));

    % Update States (Euler Method)
    x_ref(:,i) = x_ref(:,i-1) + xdot_ref * Ts;
    x(:,i) = x(:,i-1) + xdot * Ts;
    x_no_adapt(:,i) = x_no_adapt(:,i-1) + xdot_no_adapt * Ts;
    kx(i,:) = kx(i-1,:) + (kx_dot)' * Ts;
    kr(i) = kr(i-1) + kr_dot * Ts;
    theta_est(:,i) = theta_est(:,i-1) + theta_dot * Ts;

    % Plot Results 
    if mod(i, 5e1) == 0 || i == length(t)
        clf;
        subplot(2,2,[1,2]); % System States
        plot(t(1:i), x_ref(1,1:i), 'r', LineWidth=3); hold on;
        plot(t(1:i), x(1,1:i), 'k', LineWidth=1.5);
        plot(t(1:i), x_no_adapt(1,1:i), 'b', LineWidth=2);
        legend('Reference', 'Adaptive Control', 'Without Adaptive Control');
        xlabel('Time (s)'); ylabel('x_1(t)'); title('System State Comparison');

        subplot(2,2,3); % Estimated Parameters
        plot(t(1:i), theta_est(1,1:i), 'r', LineWidth=2); hold on;
        plot(t(1:i), theta_est(2,1:i), 'b', LineWidth=2);
        legend('Estimated \theta_1', 'Estimated \theta_2');
        xlabel('Time (s)'); ylabel('Parameter Values'); title('Adaptive Parameter Estimation');

        subplot(2,2,4); % Control Input
        plot(t(1:i), u(1:i), 'm', LineWidth=2); hold on;
        plot(t(1:i), u_no_adapt(1:i), 'c', LineWidth=2);
        legend('Adaptive Control', 'Without Adaptive Control');
        xlabel('Time (s)'); ylabel('Control Input u(t)');
        title('Control Input Comparison');
        sgtitle('Direct MRAC Simulation for Second-Order System'); 
        drawnow;
    end
end
