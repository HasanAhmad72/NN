clc;
clear;
close all;

% Input Data (OR Gate)
P = [0 0 1 1;
     0 1 0 1];

% Target Output
T = [0 1 1 1];

% Create Perceptron
net = perceptron;

% Train Network
[net,tr] = train(net,P,T);

% Test Network
Y = net(P);

disp('Input Data');
disp(P);

disp('Target Output');
disp(T);

disp('Predicted Output');
disp(Y);

% Plot Confusion Matrix
figure;
plotconfusion(T,Y);