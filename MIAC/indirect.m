% Indirect Model Identification Adaptive Control (MIAC) using 
% Recursive Least Squares First-Order SISO System
% Written By: Rasit
% Date: 27-Jan-2025
clc,clear,close all;

%% System Parameters (True and Reference Model)
a = 1; b = 2; theta = 0.2;  % True system parameters
am = -1; bm = 1;            % Reference model parameters

% Basis Functions
Phi = @(x) x * x;
Psi = @(x) [x; Phi(x)];

%Simulation Parameters
Ts = 1e-2;                     % Time step
t = 0:Ts:50;                   % Simulation time
ref = zeros(length(t),1);      % Input signal (initially zero)
% ref(t > 0.5) = sin(t(t > 0.5)); % Sinusoidal input after 0.5 second
ref(t > 1) = square(t(t > 1));    % Square input after 1 second

x_ref = zeros(length(t), 1);   % Reference model state
x = zeros(length(t), 1);       % Actual system state
u = zeros(length(t), 1);       % Control input
xdot = 0;

% Estimated Parameters
a_est = zeros(length(t), 1); a_est(1) = a;
theta_est = zeros(length(t), 1);
Omega_est = zeros(2, length(t));

% Without Adaptive Control
x_no_adapt = zeros(length(t), 1);
e_no_adapt = zeros(length(t), 1);

% Recursive Least Squares Initialization
R = 10 * eye(2);
r11 = zeros(length(t), 1); r11(1) = R(1,1);
r12 = zeros(length(t), 1);
r21 = zeros(length(t), 1); r21(1) = R(2,1);
r22 = zeros(length(t), 1);

%% Adaptive Control Gains
gam_a = 10;
gam_f = 10;
P = 1;
Gamma = [gam_a 0; 0 gam_f];  % Gradient Least Squares Gain

% True Control Gains
kx_true = (am - a) / b;
kr_true = bm / b;

%% Simulation Loop
figure('units', 'normalized', 'outerposition', [0 0 1 1], 'color', 'w')

for i = 2:length(t)
    if (i*Ts) > 25
        ref(i) = cos(i*Ts);
    end
    e = x_ref(i-1) - x(i-1);

    % Control Law
    kx = (am - a_est(i-1)) / b;
    kr = bm / b;
    ubar = kx * x(i-1) + kr * ref(i);
    xdot_des = a_est(i-1) * x(i-1) + b * ubar;
    epsilon = xdot_des - xdot;

    u(i-1) = kx * x(i-1) + kr * ref(i) - theta_est(i-1) * Phi(x(i-1));
    u_no_adapt = kx_true * x_no_adapt(i-1) + kr_true * ref(i);

    % Adaptive Law (Recursive Least Squares)
    Rdot = -R * Psi(x(i-1)) * Psi(x(i-1))' * R;

    % Gradient Least Squares Adaptive Law
    Omegadot = -Gamma * Psi(x(i-1)) * (epsilon + e * P);

    % System Dynamics
    xdot_ref = am * x_ref(i-1) + bm * ref(i);
    xdot = a * x(i-1) + b * u(i-1) + b * theta * Phi(x(i-1));
    xdot_no_adapt = a * x_no_adapt(i-1) + b * u_no_adapt + b * theta * Phi(x_no_adapt(i-1));

    % Update Variables (Euler Method)
    x_ref(i) = x_ref(i-1) + xdot_ref * Ts;
    x(i) = x(i-1) + xdot * Ts;
    Omega_est(:,i) = Omega_est(:,i-1) + Omegadot * Ts;
    theta_est(i) = Omega_est(2,i) / b;
    R = R + Rdot * Ts;
    a_est(i) = Omega_est(1,i);
    x_no_adapt(i) = x_no_adapt(i-1) + xdot_no_adapt * Ts;
    e_no_adapt(i) = x_ref(i) - x_no_adapt(i);

    r11(i) = R(1,1);
    r12(i) = R(1,2);
    r21(i) = R(2,1);
    r22(i) = R(2,2);

    % Plot Result
    if mod(i,5e1) == 0 || mod(i,length(t)) == 0
        clf
        subplot(2,2,[1,2]); % System State Comparison
        plot(t(1:i), x_ref(1:i), 'r', 'LineWidth', 2); hold on;
        plot(t(1:i), x(1:i), 'k', 'LineWidth', 1.5);
        plot(t(1:i), x_no_adapt(1:i), 'b', 'LineWidth', 2);
        legend('Reference', 'Adaptive Control', 'Without Adaptive Control');
        xlabel('Time (s)'); ylabel('x(t)'); title('System State Comparison');

        subplot(223); % Estimated Parameters
        plot(t(1:i), a_est(1:i), 'r', 'LineWidth', 2); hold on;
        plot(t(1:i), theta_est(1:i), 'g', 'LineWidth', 2);
        legend('Estimated a', 'Estimated \theta');
        xlabel('Time (s)'); ylabel('Parameter Values'); title('Adaptive Parameter Estimation');

        subplot(224); % Recursive Least Squares Elements
        plot(t(1:i), r11(1:i), 'b', 'LineWidth', 1.5); hold on;
        plot(t(1:i), r12(1:i), 'k', 'LineWidth', 1.5);
        plot(t(1:i), r21(1:i), 'r', 'LineWidth', 1.5);
        plot(t(1:i), r22(1:i), 'm', 'LineWidth', 1.5);
        legend('R_{11}', 'R_{12}', 'R_{21}', 'R_{22}');
        xlabel('Time (s)'); ylabel('R Values');
        title('Recursive Least Squares Elements');

        sgtitle('Indirect MIAC Simulation');
        drawnow
    end

end


