clc;
clear;
close all;

% Input Data
P = [0 0 1 1;
     0 1 0 1];

% Target Data
T = [0 1 1 1];

% Create Network
net = feedforwardnet(5);

% Use sigmoid in output layer
net.layers{2}.transferFcn = 'logsig';

% Train Network
[net,tr] = train(net,P,T);

% Test Network
Y = net(P);

disp('Input Data');
disp(P);

disp('Target Output');
disp(T);

disp('Network Output');
disp(Y);

disp('Binary Output');
disp(round(Y));

% Performance Plot
figure;
plotperform(tr);