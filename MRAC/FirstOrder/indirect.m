% Indirect Model Reference Adaptive Control (MRAC) for First-Order SISO System
% Written By: Rasit
% Date: 25-Jan-2025
clc,clear,close all;
%% System Parameters (True and Reference Model)
a = 3; b = 5; theta = 0.2;  % True system parameters (known except theta)
am = -1; bm = 4;            % Reference model parameters

%% Simulation Parameters
Ts = 1e-2;                   % Time step
t = 0:Ts:50;                 % Simulation time 
ref = zeros(length(t),1);      % Input signal (initially zero)
% ref(t > 0.5) = sin(t(t > 0.5)); % Sinusoidal input after 0.5 second
ref(t > 1) = square(t(t > 1));    % Square input after 1 second

% Initial conditions
x_ref = zeros(length(t),1); % Reference model state
x = zeros(length(t),1);     % Actual system state (adaptive)
x_no_adapt = zeros(length(t),1); % System without adaptive control

% Control inputs
u = zeros(length(t),1);     
u_no_adapt = zeros(length(t),1); 

% Estimated parameters (initial values)
a_est = a * ones(length(t),1);
b_est = b * ones(length(t),1);
theta_est = 0.1 * ones(length(t),1);

%% Adaptive Control Gains
gamma_a = 0.1; gamma_b = 0.1; gamma_theta = 0.6;
b_min = 0.1;  % Minimum value for b

% True control gains
kx_true = (am - a) / b;
kr_true = bm / b;

%% Simulation Loop
figure('units', 'normalized', 'outerposition', [0 0 1 1], 'color', 'w')
for i = 2:length(t)
    if (i*Ts) > 25
        ref(i) = cos(i*Ts);
    end
    % Control Law
    kx = (am - a_est(i-1)) / b_est(i-1);
    kr = bm / b_est(i-1);
    u(i-1) = kx * x(i-1) + kr * ref(i-1) - theta_est(i-1) * x(i-1);
    u_no_adapt(i-1) = kx_true * x_no_adapt(i-1) + kr_true * ref(i-1);

    % Adaptive Parameter Update
    e = x_ref(i-1) - x(i-1);
    a_dot = -gamma_a * x(i-1) * e;
    
    if b_est(i-1) >= b_min
        b_dot = -gamma_b * u(i-1) * e;
        if (b_est(i-1) == b_min && b_dot < 0)
            b_dot = 0;
        end
    end
    
    theta_dot = -gamma_theta * x(i-1) * e * sign(b);
    
    % System Dynamics
    xdot_ref = am * x_ref(i-1) + bm * ref(i-1);
    xdot = a * x(i-1) + b * u(i-1) + b * theta * x(i-1);
    xdot_no_adapt = a * x_no_adapt(i-1) + b * u_no_adapt(i-1) + b * theta * x_no_adapt(i-1);
    
    % Update Variables (Euler Method)
    x_ref(i) = x_ref(i-1) + xdot_ref * Ts;
    x(i) = x(i-1) + xdot * Ts;
    x_no_adapt(i) = x_no_adapt(i-1) + xdot_no_adapt * Ts;
    
    a_est(i) = a_est(i-1) + a_dot * Ts;
    b_est(i) = b_est(i-1) + b_dot * Ts;
    theta_est(i) = theta_est(i-1) + theta_dot * Ts;

    % Plot Result
    if mod(i,5e1) == 0 || mod(i,length(t)) == 0
        clf
        subplot(2,2,[1,2]); % System States
        plot(t(1:i), x_ref(1:i), 'r', LineWidth=3); hold on;
        plot(t(1:i), x(1:i), 'k', LineWidth=1.5);
        plot(t(1:i), x_no_adapt(1:i), 'b', LineWidth=2);
        legend('Reference', 'Adaptive Control', 'Without Adaptive Control');
        xlabel('Time (s)'); ylabel('x(t)'); title('System State Comparison');

        subplot(223); % Estimated Parameters
        plot(t(1:i), a_est(1:i), 'r', LineWidth=2); hold on;
        plot(t(1:i), b_est(1:i), 'b', LineWidth=2);
        plot(t(1:i), theta_est(1:i), 'g', LineWidth=2);
        legend('Estimated a-est', 'Estimated b-est', 'Estimated \theta');
        xlabel('Time (s)'); ylabel('Parameter Values'); title('Adaptive Parameter Estimation');

        subplot(224); % Control Input
        plot(t(1:i), u(1:i), 'm', LineWidth=2); hold on;
        plot(t(1:i), u_no_adapt(1:i), 'c', LineWidth=2);
        legend('Adaptive Control', 'Without Adaptive Control');
        xlabel('Time (s)'); ylabel('Control Input u(t)');
        title('Control Input Comparison');
        sgtitle('Indirect MRAC Simulation'); 
        drawnow
    end
end

