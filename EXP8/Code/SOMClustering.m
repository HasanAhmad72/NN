clc;
clear;
close all;

%% Step 1: Generate clustered data
rng(10); % For reproducibility

% Cluster 1
C1 = randn(100,2) * 0.5 + [2 2];

% Cluster 2
C2 = randn(100,2) * 0.5 + [7 3];

% Cluster 3
C3 = randn(100,2) * 0.5 + [4 8];

% Combine all data
data = [C1; C2; C3];

% SOM requires observations as columns
X = data';

%% Step 2: Plot original data
figure;

scatter(C1(:,1), C1(:,2), 40, 'r', 'filled');
hold on;

scatter(C2(:,1), C2(:,2), 40, 'g', 'filled');
scatter(C3(:,1), C3(:,2), 40, 'b', 'filled');

xlabel('Feature 1');
ylabel('Feature 2');
title('Original Data');

legend('Cluster 1','Cluster 2','Cluster 3');
grid on;

%% Step 3: Create Self-Organizing Map
gridSize = [5 5];

net = selforgmap(gridSize);

%% Step 4: Train SOM
net.trainParam.epochs = 200;

[net,tr] = train(net,X);

%% Step 5: Find Best Matching Unit
outputs = net(X);

[~, neuronIndex] = max(outputs, [], 1);

%% Step 6: Display SOM topology
figure;

plotsomtop(net);

title('SOM Network Topology');

%% Step 7: Display SOM sample hits
figure;

plotsomhits(net,X);

title('SOM Sample Hits');

%% Step 8: Display SOM weight positions
figure;

plotsompos(net,X);

title('SOM Weight Positions');

%% Step 9: Display cluster assignments
disp('Neuron assignment for first 20 samples:');

disp(neuronIndex(1:20));