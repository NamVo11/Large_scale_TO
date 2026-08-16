clear;
clc;
close all;

%% =========================================================
%  Common time settings
% ==========================================================
Tmax  = 0.12;
Nstep = 200;

dt   = Tmax / Nstep;
time = (0:Nstep) * dt;


%% =========================================================
%  Define BC_Fx
% ==========================================================
BC_Fx.type = 'sum_sin';
BC_Fx.amp  = 40000;

BC_Fx.freq_factor = 0.35;
BC_Fx.freq  = [25.0 55.0 95.0 150.0 180.0 200.0];
BC_Fx.scale = [1.00 0.82 0.58 0.44 0.30 0.30];
BC_Fx.phi   = [0.37 2.11 4.38 1.42 5.83 2.84];
BC_Fx.rise_time = 0.001;


%% =========================================================
%  Define BC_Fz
% ==========================================================
BC_Fz.type = 'sum_sin';
BC_Fz.amp  = -40000;

BC_Fz.freq_factor = 0.35;
BC_Fz.freq  = [25.0 55.0 95.0 150.0 180.0 200.0];
BC_Fz.scale = [1.00 0.82 0.58 0.44 0.30 0.30];

% Different phase from Fx
BC_Fz.phi = 0.8*[0.37 2.11 4.38 1.42 5.83 2.84];

BC_Fz.rise_time = 0.001;


%% =========================================================
%  Define BC_Fmass
%
%  +z : upward
%  -z : downward
%
%  Therefore negative Fmass means downward force.
% ==========================================================
BC_Fmass.type = 'constant';
BC_Fmass.amp  = -40000;


%% =========================================================
%  Allocate
% ==========================================================
Fx_hist    = zeros(Nstep + 1, 1);
Fz_hist    = zeros(Nstep + 1, 1);
Fmass_hist = zeros(Nstep + 1, 1);

Fsum_x = zeros(Nstep + 1, 1);
Fsum_z = zeros(Nstep + 1, 1);


%% =========================================================
%  Time loop
% ==========================================================
for istep = 0:Nstep

    id = istep + 1;

    Fx    = evaluate_bc(istep, Tmax, Nstep, BC_Fx);
    Fz    = evaluate_bc(istep, Tmax, Nstep, BC_Fz);
    Fmass = evaluate_bc(istep, Tmax, Nstep, BC_Fmass);

    Fx_hist(id)    = Fx;
    Fz_hist(id)    = Fz;
    Fmass_hist(id) = Fmass;

    % Resultant force
    Fsum_x(id) = Fx;
    Fsum_z(id) = Fz + Fmass;

end


%% =========================================================
%  Magnitude
% ==========================================================
Fmag = hypot(Fsum_x, Fsum_z);


%% =========================================================
%  Angle with vertical +z axis
%
%       0 deg
%        +z
%         ^
%         |
%         |
% -90 <--O--> +90
%         |
%         |
%        -z
%      +/-180
%
% Positive angle : toward +x
% Negative angle : toward -x
% ==========================================================
theta = atan2d(Fsum_x, Fsum_z);

theta(theta < 0) = theta(theta < 0) + 360;

theta(Fmag < 1e-12) = NaN;


%% =========================================================
%  Figure 1
%  Trajectory of resultant force
% ==========================================================
figure;
hold on;

% Trajectory of vector tip
plot(Fsum_x, Fsum_z, ...
    'LineWidth', 1.5);

% Draw several resultant-force vectors
nvec = 12;
ids = unique(round(linspace(1, Nstep + 1, nvec)));

quiver( ...
    zeros(size(ids)), ...
    zeros(size(ids)), ...
    Fsum_x(ids)', ...
    Fsum_z(ids)', ...
    0, ...
    'LineWidth', 1.0);

% Start point
plot(Fsum_x(1), Fsum_z(1), ...
    'o', ...
    'MarkerSize', 8, ...
    'LineWidth', 1.5);

% End point
plot(Fsum_x(end), Fsum_z(end), ...
    's', ...
    'MarkerSize', 8, ...
    'LineWidth', 1.5);

xline(0, 'k');
yline(0, 'k');

grid on;
axis equal;

xlabel('F_x');
ylabel('F_z');

title('Trajectory of resultant force F_{sum}');

legend( ...
    'F_{sum} trajectory', ...
    'F_{sum} vectors', ...
    'Start', ...
    'End', ...
    'Location', 'best');


%% =========================================================
%  Figure 2
%  Magnitude and angle vs time
% ==========================================================
figure;

subplot(2,1,1);

plot(time, Fmag, ...
    'LineWidth', 1.5);

grid on;

xlabel('Time [s]');
ylabel('|F_{sum}|');

title('Resultant force magnitude');


subplot(2,1,2);

plot(time, theta, ...
    'LineWidth', 1.5);

grid on;

xlabel('Time [s]');
ylabel('\theta [deg]');

ylim([0 360]);
yticks(0:45:360);

title('Angle of F_{sum} relative to +z axis');


%% =========================================================
%  Figure 3
%  Individual force histories
% ==========================================================
figure;
hold on;

plot(time, Fx_hist, ...
    'LineWidth', 1.3);

plot(time, Fz_hist, ...
    'LineWidth', 1.3);

plot(time, Fmass_hist, ...
    'LineWidth', 1.3);

plot(time, Fsum_x, '--', ...
    'LineWidth', 1.5);

plot(time, Fsum_z, '--', ...
    'LineWidth', 1.5);

grid on;

xlabel('Time [s]');
ylabel('Force');

legend( ...
    'F_x', ...
    'F_z', ...
    'F_{mass}', ...
    'F_{sum,x}', ...
    'F_{sum,z}', ...
    'Location', 'best');

title('Force components');


%% =========================================================
%  Local functions
% ==========================================================

function value = evaluate_bc(istep, Tmax, Nstep, BC)

    switch BC.type

        case 'constant'

            value = bc_value_constant( ...
                istep, Tmax, Nstep, BC);

        case 'sum_sin'

            value = bc_value_sum_sim( ...
                istep, Tmax, Nstep, BC);

        otherwise

            error('Unknown BC type: %s', BC.type);

    end

end


function value = bc_value_constant(istep, Tmax, Nstep, BC)

    %#ok<INUSD>

    value = BC.amp;

end


function value = bc_value_sum_sim(istep, Tmax, Nstep, BC)

    dt = Tmax / Nstep;
    t  = istep * dt;

    envelope = min(t / BC.rise_time, 1.0);

    wave = sum(BC.scale .* sin(2*pi*(BC.freq_factor * BC.freq)*t + BC.phi));

    %value = BC.amp * envelope * wave;

    value = BC.amp * wave;
end