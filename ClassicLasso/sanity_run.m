% test_Classic_Lasso_A.m
% Sanity‐check for A = [3 2; 1 5], sigma = 2, rhs = [3; -6].
% We expect the solver to compute xi = M\rhs where M = I + 2*(A*A').

clear; clc;

%% 1) Define Ainput.A
Ainput.A = [3  2;
            1  5];

%% 2) Define rhs
rhs = [3; -6];

%% 3) Build par:
%    - rr: logical mask of length n = 2 (both columns active = false,false)
%    - sigma = 2
%    - n = 2
par.rr    = [false; false];
par.sigma = 2;
par.n     = 2;

%% 4) Call the original MATLAB solver
[xi, resnrm, solve_ok] = Classic_Lasso_linsys_solver(Ainput, rhs, par);

%% 5) Manually form M = I + sigma*(A*A')
M = eye(2) + par.sigma * (Ainput.A * Ainput.A.');

%% 6) Solve directly for comparison
xi_direct = M \ rhs;

%% 7) Display results
disp('--- Classic_Lasso_linsys_solver output ---');
fprintf('xi (solver)    = [%.8f; %.8f]\n', xi(1), xi(2));
fprintf('xi (direct)    = [%.8f; %.8f]\n', xi_direct(1), xi_direct(2));
fprintf('Residual norm  = %.8e\n', resnrm);
fprintf('Solve flag     = %d\n', solve_ok);
fprintf('‖xi - xi_direct‖ = %.8e\n', norm(xi - xi_direct));
