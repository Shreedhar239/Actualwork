clc;
clear all;

% Load the system data
loaddata = load('loaddataNew.m');
linedata = load('linedataNew.m');

% Set parameters
evcs_power = 142;     % EVCS power in kW
n_particles = 30;     % Number of particles
max_iterations = 100; % Maximum iterations

% Run optimization
[best_location, min_losses] = optimize_evcs_placement(loaddata, linedata, evcs_power, n_particles, max_iterations);

% Display final results
if min_losses ~= inf
    fprintf('\nOptimization Results:\n');
    fprintf('Optimal EVCS Location: Bus %d\n', best_location);
    fprintf('System Losses with EVCS: %.2f kW\n', min_losses);
else
    fprintf('\nNo feasible solution found. Consider:\n');
    fprintf('1. Reducing EVCS power rating\n');
    fprintf('2. Relaxing constraints\n');
end

function [PL, voltage, line_loading] = calculate_loadflow(loaddata, linedata)
    % Initialize system parameters
    MVAb = 100;
    KVb = 11;
    Zb = (KVb^2)/MVAb;
    
    % Get system dimensions
    n_bus = size(loaddata, 1);
    n_branch = size(linedata, 1);
    
    % Initialize arrays
    R = zeros(n_branch, 1);
    X = zeros(n_branch, 1);
    P = zeros(n_bus, 1);
    Q = zeros(n_bus, 1);
    
    % Convert line impedances to per unit
    for i = 1:n_branch
        R(i) = linedata(i,4)/Zb;
        X(i) = linedata(i,5)/Zb;
    end
    
    % Convert power to per unit
    for i = 1:n_bus
        P(i) = loaddata(i,2)/(MVAb*1000);  % Convert kW to pu
        Q(i) = loaddata(i,3)/(MVAb*1000);  % Convert kVAR to pu
    end
    
    % Initialize voltage array
    V = ones(n_bus, 1);
    
    % Maximum iterations for load flow
    max_iter = 100;
    tolerance = 1e-6;
    
    % Forward-Backward Sweep Method
    for iter = 1:max_iter
        V_old = V;
        
        % Calculate branch currents (backward sweep)
        I_branch = zeros(n_branch, 1);
        for i = n_branch:-1:1
            from_bus = linedata(i,2);
            to_bus = linedata(i,3);
            
            % Calculate load current at receiving end
            if to_bus <= length(P)
                I_load = conj((P(to_bus) + 1j*Q(to_bus))/V(to_bus));
                I_branch(i) = I_load;
            end
            
            % Add currents from downstream branches
            for j = 1:n_branch
                if linedata(j,2) == to_bus
                    I_branch(i) = I_branch(i) + I_branch(j);
                end
            end
        end
        
        % Update voltages (forward sweep)
        V(1) = 1.0; % Slack bus voltage
        for i = 1:n_branch
            from_bus = linedata(i,2);
            to_bus = linedata(i,3);
            Z = R(i) + 1j*X(i);
            if to_bus <= length(V) && from_bus <= length(V)
                V(to_bus) = V(from_bus) - Z*I_branch(i);
            end
        end
        
        % Check convergence
        if max(abs(V - V_old)) < tolerance
            break;
        end
    end
    
    % Calculate power losses
    Pl = zeros(n_branch, 1);
    line_loading = zeros(n_branch, 1);
    Ib = (MVAb*1000)/(sqrt(3)*KVb*1000); % Base current
    
    for i = 1:n_branch
        I_mag = abs(I_branch(i));
        Pl(i) = (I_mag^2)*R(i);
        line_loading(i) = (I_mag/Ib)*100; % Convert to percentage
    end
    
    % Prepare outputs
    PL = sum(Pl)*MVAb*1000; % Convert to kW
    voltage = abs(V);
    line_loading = real(line_loading); % Ensure real values
end


function [best_position, best_fitness] = optimize_evcs_placement(loaddata, linedata, evcs_power, n_particles, max_iterations)
    % PSO parameters
    w = 0.729; % Inertia weight
    c1 = 2.05; % Cognitive parameter
    c2 = 2.05; % Social parameter
     

    
    % Problem dimensions
    n_buses = size(loaddata, 1);
    
    % Calculate base case for comparison
    [base_PL, base_V, base_loading] = calculate_loadflow(loaddata, linedata);
    fprintf('Base case results:\n');
    fprintf('Total losses: %.4f kW\n', base_PL);
    fprintf('Voltage range: %.4f - %.4f pu\n', min(base_V), max(base_V));
    fprintf('Maximum line loading: %.2f%%\n\n', max(base_loading));
    
    % Initialize particles (excluding slack bus)
    positions = randi([2, n_buses], n_particles, 1);
    velocities = zeros(n_particles, 1);
    
    % Initialize best positions and fitness
    personal_best_pos = positions;
    personal_best_fit = inf(n_particles, 1);
    global_best_pos = 0;
    global_best_fit = inf;
    
    % Main PSO loop
    for iter = 1:max_iterations
        % Evaluate each particle
        for i = 1:n_particles
            % Create modified load data with EVCS
            mod_loaddata = loaddata;
            bus_idx = positions(i);
            mod_loaddata(bus_idx, 2) = mod_loaddata(bus_idx, 2) + evcs_power;
            mod_loaddata(bus_idx, 3) = mod_loaddata(bus_idx, 3) + evcs_power*0.3; % 0.95 power factor
            
            % Run load flow
            [PL, V, loading] = calculate_loadflow(mod_loaddata, linedata);
            
            % Check constraints
            is_feasible = min(V) >= 0.95 && max(V) <= 1.05 && max(loading) <= 80;
            
            % Update fitness
            if is_feasible
                if PL < personal_best_fit(i)
                    personal_best_fit(i) = PL;
                    personal_best_pos(i) = positions(i);
                    
                    if PL < global_best_fit
                        global_best_fit = PL;
                        global_best_pos = positions(i);
                    end
                end
            end
            
            % Debug information for first particle in first iteration
            if iter == 1 && i == 1
                fprintf('Debug Information for First Particle:\n');
                fprintf('Position: Bus %d\n', positions(i));
                fprintf('Voltage range: %.4f - %.4f pu\n', min(V), max(V));
                fprintf('Maximum line loading: %.2f%%\n', max(loading));
                fprintf('Power loss: %.4f kW\n', PL);
                fprintf('Feasible: %d\n\n', is_feasible);
            end
        end
        
        % Update velocities and positions
        for i = 1:n_particles
            % Update velocity
            r1 = rand();
            r2 = rand();
            velocities(i) = w*velocities(i) + ...
                           c1*r1*(personal_best_pos(i) - positions(i)) + ...
                           c2*r2*(global_best_pos - positions(i));
            
            % Update position
            new_pos = round(positions(i) + velocities(i));
            positions(i) = max(2, min(new_pos, n_buses)); % Keep within bounds
        end
        
        % Display progress
        if global_best_pos ~= 0
            fprintf('Iteration %d: Best Loss = %.4f kW at Bus %d\n', ...
                    iter, global_best_fit, global_best_pos);
        else
            fprintf('Iteration %d: No feasible solution found yet\n', iter);
        end
    end
    
    best_position = global_best_pos;
    best_fitness = global_best_fit;
end