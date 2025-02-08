% Load Flow Analysis Code for 51-bus System
% Clear workspace and set format
clear all;
clc;
format short;

% Load input data
m = load('loaddataNew.m');
l = load('linedataNew.m');
br = length(l);
no = length(m);

% Initialize system parameters
f = 0;
d = 0;
MVAb = 100;
KVb = 11;
Zb = (KVb^2)/MVAb;

% Initialize arrays for 51-bus system
Pg = zeros(51,1);
Pg1 = zeros(51,1);
R = zeros(br,1);
X = zeros(br,1);
P = zeros(51,1);
Q = zeros(51,1);

% Convert impedances to per unit
for i=1:br
    R(i,1) = (l(i,4))/Zb;
    X(i,1) = (l(i,5))/Zb;
end

% Convert power to per unit
for i=1:no
    P(i,1) = (m(i,2)/(1000*MVAb));
    Q(i,1) = (m(i,3)/(1000*MVAb));
end

% Create connectivity matrix
C = zeros(br,51);
for i=1:br
    a = l(i,2);
    b = l(i,3);
    for j=1:no
        if a==j
            C(i,j) = -1;
        end
        if b==j
            C(i,j) = 1;
        end
    end
end

% Find end nodes
e = 1;
endnode = zeros(51,1);
for i=1:no
    d = 0;
    for j=1:br
        if C(j,i)==-1
            d = 1;
        end
    end
    if d==0
        endnode(e,1) = i;
        e = e+1;
    end
end

% Get number of end nodes
h = length(nonzeros(endnode));

% Initialize path matrix
g = zeros(51,51);

% Find paths from end nodes to source
for j=1:h
    e = 2;
    f = endnode(j,1);
    for s=1:no
        if (f~=1)
            k = 1;
            for i=1:br
                if ((C(i,f)==1)&&(k==1))
                    f = i;
                    k = 2;
                end
            end
            k = 1;
            for i=1:no
                if ((C(f,i)==-1)&&(k==1))
                    f = i;
                    g(j,e) = i;
                    e = e+1;
                    k = 3;
                end
            end
        end
    end
end

% Add end nodes to first column
for i=1:h
    g(i,1) = endnode(i,1);
end

% Get width of path matrix
w = size(g,2);

% Sort paths
for i=1:h
    j = 1;
    for k=1:no
        for t=1:w
            if g(i,t)==k
                g(i,t) = g(i,j);
                g(i,j) = k;
                j = j+1;
            end
        end
    end
end

% Initialize adjacency matrix
adjb = zeros(br,51);

% Create adjacency matrix
for k=1:br
    e = 1;
    for i=1:h
        for j=1:w-1
            if (g(i,j)==k)
                if g(i,j+1)~=0
                    adjb(k,e) = g(i,j+1);
                    e = e+1;
                else
                    adjb(k,1) = 0;
                end
            end
        end
    end
end

% Clean up adjacency matrix
for i=1:br-1
    for j=h:-1:1
        if j <= size(adjb,2)
            for k=j:-1:2
                if k <= size(adjb,2)
                    if adjb(i,j)==adjb(i,k-1)
                        adjb(i,j) = 0;
                    end
                end
            end
        end
    end
end

% Remove zeros and adjust indices
x = size(adjb,1);
ab = size(adjb,2);
for i=1:x
    for j=1:ab
        if adjb(i,j)==0 && j~=ab
            if adjb(i,j+1)~=0
                adjb(i,j) = adjb(i,j+1);
                adjb(i,j+1) = 0;
            end
        end
        if adjb(i,j)~=0
            adjb(i,j) = adjb(i,j)-1;
        end
    end
end

% Create reduced adjacency matrix
adjcb = zeros(br-1,51);
for i=1:x-1
    for j=1:ab
        adjcb(i,j) = adjb(i+1,j);
    end
end

% Initialize voltage array
vb = ones(51,1);

% Main load flow calculation loop
for s=1:10
    % Calculate node load currents
    nlc = zeros(51,1);
    for i=1:min(no,51)
        if i <= size(vb,1)
            nlc(i,1) = conj(complex(P(i,1),Q(i,1)))/(vb(i,1));
        end
    end
    
    % Initialize branch currents
    Ibr = zeros(br,1);
    for i=1:br
        if i+1 <= size(nlc,1)
            Ibr(i,1) = nlc(i+1,1);
        end
    end
    
    % Calculate branch currents
    xy = size(adjcb,2);
    for i=br-1:-1:1
        for k=1:xy
            if adjcb(i,k)~=0
                u = adjcb(i,k);
                if u <= size(Ibr,1)
                    Ibr(i,1) = Ibr(i,1) + Ibr(u,1);
                end
            end
        end
    end
    
    % Update voltages
    for i=2:no
        g = 0;
        for a=1:size(adjcb,1)
            if xy>1
                if i-1 <= size(adjcb,2)
                    if adjcb(a,2)==i-1
                        u = adjcb(a,1);
                        if u <= size(vb,1) && i-1 <= size(Ibr,1) && i-1 <= size(R,1) && i-1 <= size(X,1)
                            vb(i,1) = ((vb(u,1))-((Ibr(i-1,1))*(complex(R(i-1,1),X(i-1,1)))));
                        end
                        g = 1;
                    end
                    if adjcb(a,3)==i-1
                        u = adjcb(a,1);
                        if u <= size(vb,1) && i-1 <= size(Ibr,1) && i-1 <= size(R,1) && i-1 <= size(X,1)
                            vb(i,1) = ((vb(u,1))-((Ibr(i-1,1))*(complex(R(i-1,1),X(i-1,1)))));
                        end
                        g = 1;
                    end
                end
            end
        end
        if g==0
            if i <= size(vb,1) && i-1 <= size(vb,1) && i-1 <= size(Ibr,1) && i-1 <= size(R,1) && i-1 <= size(X,1)
                vb(i,1) = ((vb(i-1,1))-((Ibr(i-1,1))*(complex(R(i-1,1),X(i-1,1)))));
            end
        end
    end
end

% Calculate voltage magnitudes
vbp = abs(vb);
va = zeros(51,2);
for i=1:no
    va(i,2) = vbp(i,1);
    va(i,1) = i;
    P1(i) = P(i);
    Q1(i) = Q(i);
end

% Calculate branch current magnitudes
Ibrp = abs(Ibr);

% Initialize and calculate losses
PL = zeros(1,1);
QL = zeros(1,1);
Pl = zeros(br,1);
Ql = zeros(br,1);

for f=1:br
    if f <= size(Ibrp,1) && f <= size(R,1) && f <= size(X,1)
        Pl(f,1) = (Ibrp(f,1)^2)*R(f,1);
        Ql(f,1) = X(f,1)*(Ibrp(f,1)^2);
        PL(1,1) = PL(1,1) + Pl(f,1);
        QL(1,1) = QL(1,1) + Ql(f,1);
    end
end

% Convert losses to kW and kVAR
Plosskw = (Pl)*100000;
Qlosskw = (Ql)*100000;
PL = (PL)*100000;
QL = (QL)*100000;

% Get voltage profile
voltage = vbp(:,1);
v_mag = va(:,2);

% Display results
fprintf('\nNetwork Information:\n');
fprintf('Number of buses: %d\n', no);
fprintf('Number of branches: %d\n', br);
fprintf('Base MVA: %.2f\n', MVAb);
fprintf('Base kV: %.2f\n', KVb);

fprintf('\nResults Summary:\n');
disp('Branch Losses (kW):');
disp(Plosskw);
disp('Branch Reactive Losses (kVAR):');
disp(Qlosskw);
disp('Total Active Power Loss (kW):');
disp(PL);
disp('Total Reactive Power Loss (kVAR):');
disp(QL);
disp('Voltage Profile (p.u.):');
disp(voltage);

% Plot voltage profile
figure;
plot(1:length(voltage), voltage, 'b-o', 'LineWidth', 2);
grid on;
title('Voltage Profile of the System');
xlabel('Bus Number');
ylabel('Voltage (p.u.)');


% for plotting bar and formatting the graph
bus=1:1:51;
bus=bus.';
bar(bus,voltage,0.2)
xticks(bus);
xlabel('Bus');
ylabel('Voltage in pu');
ylim([0 1.1]);
% for plotting bar and formatting the graph
bus=1:1:no;
bus=bus.';
pBus= bus(2:end);
subplot(2,1,1)
bar(pBus,Plosskw);
xticks(pBus);
xticklabels({'1-2','2-3','2-4','4-5','5-6','6-7','7-8','8-9','9-10','10-11','11-12','12-13','13-14','14-15','13-16','16-17','17-18','16-19','19-20','19-21','21-22','22-23','23-24','24-25','25-26','24-27','27-28','28-29','29-30','24-31','31-32','32-33','33-34','34-35','35-36','36-37','37-38','36-39','39-40','40-41','41-42','42-43','43-44','42-45','45-46','46-47','46-48','48-49','49-50','50-51'})
xlabel('Branch');
ylabel('Active Power loss (Kw)');
ylim([0 1.1]);
subplot(2,1,2)
figure()
bar(bus,voltage,0.5);
xticks(bus);
xlabel('Bus');
ylabel('Voltage in pu');
ylim([0 1.1]);
figure()
bar(pBus,Qlosskw);
xticks(pBus);
xticklabels({'1-2','2-3','2-4','4-5','5-6','6-7','7-8','8-9','9-10','10-11','11-12','12-13','13-14','14-15','13-16','16-17','17-18','16-19','19-20','19-21','21-22','22-23','23-24','24-25','25-26','24-27','27-28','28-29','29-30','24-31','31-32','32-33','33-34','34-35','35-36','36-37','37-38','36-39','39-40','40-41','41-42','42-43','43-44','42-45','45-46','46-47','46-48','48-49','49-50','50-51'})
xlabel('Branch');
ylabel('Reactive Power loss (KVAR)');



















