clc;
clear;
close all;

% Discrete Hopfield Network
% Stored Pattern = [1 1 1 0]

%% Step 1: Define the stored pattern
pattern = [1 1 1 0];

% Convert binary pattern to bipolar pattern
% 0 -> -1, 1 -> +1
bipolar_pattern = 2 * pattern - 1;

disp('Stored Pattern:');
disp(pattern);

disp('Bipolar Stored Pattern:');
disp(bipolar_pattern);

%% Step 2: Calculate Weight Matrix
W = bipolar_pattern' * bipolar_pattern;

% Set diagonal elements to zero
W = W - diag(diag(W));

disp('Weight Matrix:');
disp(W);

%% Step 3: Define input pattern with mistakes
% Mistakes are introduced in first and second positions
test_pattern = [0 0 1 0];

% Convert test pattern to bipolar form
test_bipolar = 2 * test_pattern - 1;

disp('Input Pattern with Mistakes:');
disp(test_pattern);

disp('Bipolar Input:');
disp(test_bipolar);

%% Step 4: Recall the stored pattern
x = test_bipolar';

net = W * x;

% Apply bipolar activation function
output = sign(net);

output(output == 0) = 1;

%% Step 5: Convert bipolar output back to binary
final_output = (output' + 1) / 2;

disp('Net Input:');
disp(net');

disp('Recovered Bipolar Pattern:');
disp(output');

disp('Recovered Binary Pattern:');
disp(final_output);

%% Step 6: Compare with stored pattern
if isequal(final_output, pattern)
    disp('Stored pattern successfully recalled.');
else
    disp('Pattern is not completely recalled.');
end