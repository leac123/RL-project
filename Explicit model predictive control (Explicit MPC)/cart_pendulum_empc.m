yalmip('clear')
clear all

%% Cart pendulum model
m_c = 1.0;      % Cart mass
m_p = 0.1;      % Pendulum mass
l = 1.0;        % Pendulum length
g = 9.81;       % Gravity

% The linearized model about the equilibrium

A = [0, 0, 1, 0;
     0, 0, 0, 1;
     0, -g*m_p/m_c, 0, 0;
     0, g*(m_c + m_p)/(l*m_c), 0, 0];

B = [0; 0; 1/m_c; -1/(l*m_c)];

% The discrete time linear model
Ts = 0.2;
sys = c2d(ss(A, B, [], []), Ts, 'zoh');

Ad = sys.A;

Bd = sys.B;


%% Model predictive control - Explicit multi-parametric solution

nx = 4; % Number of states
nu = 1; % Number of inputs

% Prediction horizon
N = 15;

% Weighting matrices
Q = diag([1, 10, 1, 1]);
R = 1;

% State bounds
bound = [2; 0.5; 3; pi];

% States x(k), ..., x(k+N)
x = sdpvar(repmat(nx,1,N),repmat(1,1,N));
% Inputs u(k), ..., u(k+N) (last one not used)
u = sdpvar(repmat(nu,1,N),repmat(1,1,N));
% Binary for PWA selection
d = binvar(repmat(4,1,N),repmat(1,1,N));

constraints = [];
objective = 0;

for k = N-1:-1:1   

    % Feasible region
    constraints = [constraints , -20    <= u{k}   <= 20,
                                 -bound <= x{k}   <= bound,
                                 -bound <= x{k+1} <= bound];
    % PWA Dynamics
    constraints = [constraints ,x{k+1} == Ad*x{k}+Bd*u{k}];

    % Add stage cost to total cost
    objective = objective + x{k}'*Q*x{k} + u{k}'*R*u{k};
end

[sol,diagn,Z,Valuefcn,Optimizer] = solvemp(constraints,objective ,[],x{1},u{1});

%% Simulate
Ts_sim = 0.01;
sim_time = 500; 
x0 = [0; 0.3; 0; 0];
x_vec = zeros(nx, sim_time);
t_vec = 0:Ts_sim:(sim_time-1)*Ts_sim;
x_vec(:, 1) = x0;
u_vec = zeros(nu, sim_time);
for i = 1:sim_time - 1
    assign(x{1},x_vec(:, i));
    u_t = value(Optimizer);
    u_vec(i) = u_t;
    x_vec(:, i + 1) = x_vec(:, i) + Ts_sim*(A*x_vec(:, i) + B*u_t);
end

figure; plot(t_vec, x_vec);
figure; plot(t_vec, u_vec);

%% Save solution to file
pwa_u = struct('Fi', {sol{1}.Fi}, 'Gi', {sol{1}.Gi}, 'Hi', {[]}, 'Ki', {[]});
for i = 1:length(sol{1}.Fi)
    [H, K] = double(sol{1}.Pn(i));
    pwa_u.Hi{i} = H;
    pwa_u.Ki{i} = K;
end
save('empc_sol.mat', 'pwa_u')