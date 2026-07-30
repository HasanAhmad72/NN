clc;
clear;
close all;

% Input combinations
inputs = [0 0;
          0 1;
          1 0;
          1 1];

disp('======================================');
disp(' McCulloch-Pitts Model Logic Gates');
disp('======================================');

%% AND Gate
disp(' ');
disp('AND Gate');

w = [1 1];
theta = 2;

for i = 1:4
    net = inputs(i,:) * w';
    if net >= theta
        y = 1;
    else
        y = 0;
    end
    fprintf('%d %d -> %d\n', inputs(i,1), inputs(i,2), y);
end

%% OR Gate
disp(' ');
disp('OR Gate');

w = [1 1];
theta = 1;

for i = 1:4
    net = inputs(i,:) * w';
    if net >= theta
        y = 1;
    else
        y = 0;
    end
    fprintf('%d %d -> %d\n', inputs(i,1), inputs(i,2), y);
end

%% NOT Gate
disp(' ');
disp('NOT Gate');

input_not = [0;1];
w = -1;
theta = 0;

for i = 1:2
    net = input_not(i) * w;
    if net >= theta
        y = 1;
    else
        y = 0;
    end
    fprintf('%d -> %d\n', input_not(i), y);
end

%% XOR Gate (2-Layer McCulloch-Pitts Network)
disp(' ');
disp('XOR Gate');

for i = 1:4

    A = inputs(i,1);
    B = inputs(i,2);

    % Hidden Layer
    OR_out  = (A + B) >= 1;   % OR neuron
    AND_out = (A + B) >= 2;   % AND neuron

    % Output Layer
    % XOR = OR AND NOT(AND)
    if (OR_out == 1 && AND_out == 0)
        XOR_out = 1;
    else
        XOR_out = 0;
    end

    fprintf('%d %d -> %d\n', A, B, XOR_out);

end