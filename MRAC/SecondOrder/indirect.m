% Indirect MRAC for Second-Order SISO System
% Based on Dr. Nhan Nguyen's MRAC (Section 5.5)
% Updated for Consistency by Joe Chai (2019) & Rasit (2025)

clc, clear, close all;

%% System Parameters (True and Reference Model)
w = 1; 
zeta = -0.5; 
b = 1;

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

w_est = ones(length(t), 1); 
zeta_est = -0.5 * ones(length(t), 1);
b_est = ones(length(t), 1);

theta_est = zeros(2, length(t));

%% Adaptive Control Gains
gamma_w = 1;
gamma_z = 1;
gamma_b = 1;
gamma_theta = diag([30 30]);

w0 = 0.02;
b0 = 0.1;

% True Control Gains (for non-adaptive comparison)
kx_true = pinv(B' * B) * B' * (Am - A);
kr_true = bm / b;

%% Simulation Loop
figure('units', 'normalized', 'outerposition', [0 0 1 1], 'color', 'w');
gif('IndirectMRAC.gif')
for i = 2:length(t)
    if (i*Ts) > 25
        ref(i) = cos(i*Ts);
    end

    % Estimated System Model
    B_est = [0; b_est(i-1)];
    A_est = [0 1; -w_est(i-1)^2 -2*zeta_est(i-1)*w_est(i-1)];

    % Compute Adaptive Control Gains
    kx = pinv(B_est' * B_est) * B_est' * (Am - A_est);
    kr = bm / b_est(i-1);

    % Control Law
    u(i-1) = kx * x(:,i-1) + kr * ref(i) - theta_est(:,i-1)' * phi(x(:,i-1));
    u_no_adapt(i-1) = kx_true * x_no_adapt(:,i-1) + kr_true * ref(i);

    % Adaptive Parameter Update
    u_bar = kx * x(:,i-1) + kr * ref(i);
    e = x_ref(:,i-1) - x(:,i-1);

    % Frequency Parameter Adaptation
    if w_est(i-1) >= w0
        w_dot = gamma_w * x(1,i-1) * e' * Pbar / (2 * w_est(i-1));
        if (w_est(i-1) == w0 && w_dot < 0), w_dot = 0; end
    else
        w_dot = 0;
    end

    % Damping Parameter Adaptation
    zeta_dot = (gamma_z * x(2,i-1) * w_est(i-1) - gamma_w * x(1,i-1) * zeta_est(i-1)) * e' * Pbar / (2 * w_est(i-1)^2);

    % Control Gain Adaptation
    if b_est(i-1) >= b0
        b_dot = -gamma_b * u_bar * e' * Pbar;
        if (b_est(i-1) == b0 && b_dot < 0), b_dot = 0; end
    else
        b_dot = 0;
    end

    % Theta Adaptation
    theta_dot = -gamma_theta * phi(x(:,i-1)) * e' * Pbar * sign(b);

    % System Dynamics
    xdot_ref = Am * x_ref(:,i-1) + Bm * ref(i);
    xdot = A * x(:,i-1) + B * u(i-1) + B * theta' * phi(x(:,i-1));
    xdot_no_adapt = A * x_no_adapt(:,i-1) + B * u_no_adapt(i-1) + B * theta' * phi(x_no_adapt(:,i-1));

    % Update States (Euler Method)
    x_ref(:,i) = x_ref(:,i-1) + xdot_ref * Ts;
    x(:,i) = x(:,i-1) + xdot * Ts;
    x_no_adapt(:,i) = x_no_adapt(:,i-1) + xdot_no_adapt * Ts;
    
    w_est(i) = w_est(i-1) + w_dot * Ts;
    zeta_est(i) = zeta_est(i-1) + zeta_dot * Ts;
    b_est(i) = b_est(i-1) + b_dot * Ts;
    theta_est(:,i) = theta_est(:,i-1) + theta_dot * Ts;
    
    % Plot Results 
    if mod(i, 5e1) == 0 || i == length(t)
        subplot(2,2,[1,2]); % System States
        plot(t, x_ref(1,:), 'b--', LineWidth=2); hold on;
        plot(t, x(1,:), 'r-', LineWidth=1.5);
        plot(t, x_no_adapt(1,:), 'k:', LineWidth=1.5);
        legend('Reference', 'Adaptive', 'No Adapt');
        xlabel('Time (s)'); ylabel('x_1(t)');
        title('System Response');
        
        subplot(2,2,3);
        plot(t, theta_est(1,:), 'r', LineWidth=1.5); hold on;
        plot(t, theta_est(2,:), 'b', LineWidth=1.5);
        legend('\theta_1', '\theta_2');
        xlabel('Time (s)'); ylabel('Parameter Estimates');
        title('Adaptive Parameter Estimates');
        
        subplot(2,2,4); 
        plot(t, w_est, 'b', LineWidth=1.5); hold on;
        plot(t, b_est, 'k', LineWidth=1.5);
        plot(t, zeta_est, 'r', LineWidth=1.5);
        legend('w', 'b', '\zeta');
        xlabel('Time (s)'); ylabel('Estimated Values');
        title('Estimated System Parameters');
        sgtitle('Indirect MRAC Simulation for Second-Order System'); 
        drawnow;
    end
end


