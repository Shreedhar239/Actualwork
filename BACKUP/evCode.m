clc
clear all;
% Load data
loaddata = load('loaddataNew.m');
linedata = load('linedataNew.m');

% Set parameters
evcs_power = 142;     % EVCS power in kW
n_particles = 30;     % Number of particles
max_iterations = 100; % Maximum iterations

% Run optimization with three-zone considerations
[best_location, min_losses] = optimize_evcs_placement(loaddata, linedata, evcs_power, n_particles, max_iterations);

function [best_position, best_fitness] = optimize_evcs_placement(loaddata, linedata, evcs_power, n_particles, max_iterations)
    % PSO parameters
    w = 0.729; 
    c1 = 2.05; 
    c2 = 2.05; 
    
    % Problem dimensions
    n_buses = size(loaddata, 1);
    
    % Define zones
    urban_zones = [10,24,32];        % High load density areas
    semi_urban_zones = [22,35,39,41,45,51];   % Medium load density areas
    rural_zones = setdiff(2:n_buses, [urban_zones, semi_urban_zones]);
    
    % Create zone weights
    zone_weights = ones(n_buses, 1) * 0.3;    % Default rural weight
    zone_weights(semi_urban_zones) = 0.6;      % Medium weight for semi-urban
    zone_weights(urban_zones) = 0.9;           % Highest weight for urban
    
    % Define minimum load requirements for each zone
    min_load_urban = 30;      % kW
    min_load_semi_urban = 20; % kW
    min_load_rural = 10;      % kW
    
    % Store zone information in a structure
    zone_info.urban = urban_zones;
    zone_info.semi_urban = semi_urban_zones;
    zone_info.rural = rural_zones;
    zone_info.min_load_urban = min_load_urban;
    zone_info.min_load_semi_urban = min_load_semi_urban;
    zone_info.min_load_rural = min_load_rural;
    
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
            [fitness, is_feasible, metrics] = evaluate_fitness_with_zones(positions(i), loaddata, linedata, ...
                                                            evcs_power, zone_weights, base_PL, zone_info);
            
            
            % Update personal best if feasible and better
            if is_feasible && fitness < personal_best_fit(i)
                personal_best_fit(i) = fitness;
                personal_best_pos(i) = positions(i);
                
                % Update global best
                if fitness < global_best_fit
                    global_best_fit = fitness;
                    global_best_pos = positions(i);
                    
                    % Print detailed metrics for new best solution
                    print_best_solution_metrics(positions(i), metrics, zone_info);
                end
            end
            
            % Debug information for first particle in first iteration
            if iter == 1 && i == 1
                fprintf('Debug Information for First Particle:\n');
                fprintf('Position: Bus %d\n', positions(i));
                zone_type = get_zone_type(positions(i), zone_info);
                fprintf('Zone: %s\n', zone_type);
                fprintf('Zone Weight: %.2f\n', zone_weights(positions(i)));
                fprintf('Metrics: Loss=%.4f kW, V_min=%.4f, Loading_max=%.2f%%\n\n', ...
                    metrics.power_loss, metrics.min_voltage, metrics.max_loading);
            end
        end
        
        % Update velocities and positions
        for i = 1:n_particles
            r1 = rand();
            r2 = rand();
            velocities(i) = w*velocities(i) + ...
                           c1*r1*(personal_best_pos(i) - positions(i)) + ...
                           c2*r2*(global_best_pos - positions(i));
            
            % Update position
            new_pos = round(positions(i) + velocities(i));
            positions(i) = max(2, min(new_pos, n_buses));
        end
        
        % Display progress
        if global_best_pos ~= 0
            zone_type = get_zone_type(global_best_pos, zone_info);
            fprintf('Iteration %d: Best Loss = %.4f kW at Bus %d (%s zone)\n', ...
                    iter, global_best_fit, global_best_pos, zone_type);
        else
            fprintf('Iteration %d: No feasible solution found yet\n', iter);
        end
    end
    
    best_position = global_best_pos;
    best_fitness = global_best_fit;
    
    % Final results
    print_final_results(best_position, best_fitness, zone_info);
end

function [fitness, is_feasible, metrics] = evaluate_fitness_with_zones(position, loaddata, linedata, ...
                                                        evcs_power, zone_weights, base_PL, zone_info)
    % Create modified load data with EVCS
    mod_loaddata = loaddata;
    mod_loaddata(position, 2) = mod_loaddata(position, 2) + evcs_power;
    mod_loaddata(position, 3) = mod_loaddata(position, 3) + evcs_power*0.3;
    
    % Run load flow
    [PL, V, loading] = calculate_loadflow(mod_loaddata, linedata);
   
    
    % Store metrics
    metrics.power_loss = PL;
    metrics.min_voltage = min(V);
    metrics.max_voltage = max(V);
    metrics.max_loading = max(loading);
    
    % Check voltage and loading constraints
    voltage_violated = any(V < 0.95 | V > 1.05);
    loading_violated = any(loading > 80);
    
    % Check zone-specific load requirements
    existing_load = loaddata(position, 2);
    zone_type = get_zone_type(position, zone_info);
    
    switch zone_type
        case 'Urban'
            load_sufficient = existing_load >= zone_info.min_load_urban;
        case 'Semi-Urban'
            load_sufficient = existing_load >= zone_info.min_load_semi_urban;
        case 'Rural'
            load_sufficient = existing_load >= zone_info.min_load_rural;
    end
    
    is_feasible = ~voltage_violated && ~loading_violated && load_sufficient;
    
    if is_feasible
        % Calculate normalized power loss
        norm_loss = PL/base_PL;
        
        % Combined fitness function with zone weighting
        w_loss = 0.6;    % Weight for power loss
        w_zone = 0.4;    % Weight for zone consideration
        
        fitness = w_loss * norm_loss - w_zone * zone_weights(position);
    else
        fitness = inf;
    end
end

function zone_type = get_zone_type(position, zone_info)
    if ismember(position, zone_info.urban)
        zone_type = 'Urban';
    elseif ismember(position, zone_info.semi_urban)
        zone_type = 'Semi-Urban';
    else
        zone_type = 'Rural';
    end
end

function print_best_solution_metrics(position, metrics, zone_info)
    zone_type = get_zone_type(position, zone_info);
    fprintf('\nNew Best Solution Metrics:\n');
    fprintf('Location: Bus %d (%s zone)\n', position, zone_type);
    fprintf('Power Loss: %.4f kW\n', metrics.power_loss);
    fprintf('Voltage Range: %.4f - %.4f pu\n', metrics.min_voltage, metrics.max_voltage);
    fprintf('Maximum Line Loading: %.2f%%\n', metrics.max_loading);
    
    % Print zone-specific metrics
    fprintf('Zone Characteristics:\n');
    switch zone_type
        case 'Urban'
            fprintf('Minimum Load Requirement: %.2f kW\n', zone_info.min_load_urban);
        case 'Semi-Urban'
            fprintf('Minimum Load Requirement: %.2f kW\n', zone_info.min_load_semi_urban);
        case 'Rural'
            fprintf('Minimum Load Requirement: %.2f kW\n', zone_info.min_load_rural);
    end
    fprintf('\n');
end

function print_final_results(best_position, best_fitness, zone_info)
    zone_type = get_zone_type(best_position, zone_info);
    fprintf('\nFinal Optimization Results:\n');
    fprintf('Optimal EVCS Location: Bus %d\n', best_position);
    fprintf('Zone Type: %s\n', zone_type);
    fprintf('Final Fitness Value: %.4f\n', best_fitness);
end

%%
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