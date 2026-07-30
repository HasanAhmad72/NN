clc;
clear;
close all;

% Input values
x = -10:0.1:10;

% Weight and Bias
w = 1;
b = 0;

% Weighted Sum
net = w*x + b;

% Sigmoid Activation Function
y = 1 ./ (1 + exp(-net));

% Display sample values
disp(' Input      Output');
disp([x(1:10:end)' y(1:10:end)']);

% Plot
figure;
plot(x,y,'b','LineWidth',2);
grid on;
xlabel('Input');
ylabel('Neuron Output');
title('Artificial Neuron using Sigmoid Activation');