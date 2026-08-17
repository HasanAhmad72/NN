clc;
clear;
close all;

%% XOR Function Using Backpropagation with Momentum

% XOR training data
X = [0 0;
     0 1;
     1 0;
     1 1];

% Target outputs
T = [0;
     1;
     1;
     0];

%% Network Architecture
% 2 Input Neurons -> 2 Hidden Neurons -> 1 Output Neuron

inputNeurons = 2;
hiddenNeurons = 2;
outputNeurons = 1;

%% Learning Parameters
learningRate = 0.5;
momentum = 0.9;
maxEpochs = 10000;
targetMSE = 0.001;

%% Initialize Weights and Biases
rng(1);

W_ih = randn(inputNeurons, hiddenNeurons) * 0.5;
b_h = randn(1, hiddenNeurons) * 0.5;

W_ho = randn(hiddenNeurons, outputNeurons) * 0.5;
b_o = randn(1, outputNeurons) * 0.5;

%% Previous Weight Changes for Momentum
dW_ih_prev = zeros(size(W_ih));
db_h_prev = zeros(size(b_h));

dW_ho_prev = zeros(size(W_ho));
db_o_prev = zeros(size(b_o));

%% Sigmoid Function
sigmoid = @(x) 1 ./ (1 + exp(-x));

%% Training
MSE_history = zeros(maxEpochs, 1);

for epoch = 1:maxEpochs

    totalError = 0;

    for i = 1:size(X,1)

        %% Forward Propagation

        % Hidden layer
        net_h = X(i,:) * W_ih + b_h;
        h = sigmoid(net_h);

        % Output layer
        net_o = h * W_ho + b_o;
        y = sigmoid(net_o);

        %% Error
        error = T(i) - y;

        totalError = totalError + error^2;

        %% Output Delta
        delta_o = error * y * (1 - y);

        %% Hidden Layer Delta
        delta_h = (h .* (1 - h)) .* (delta_o * W_ho');

        %% Weight Updates with Momentum

        % Hidden to Output
        dW_ho = learningRate * (h' * delta_o) ...
                + momentum * dW_ho_prev;

        db_o = learningRate * delta_o ...
               + momentum * db_o_prev;

        % Input to Hidden
        dW_ih = learningRate * (X(i,:)' * delta_h) ...
                + momentum * dW_ih_prev;

        db_h = learningRate * delta_h ...
               + momentum * db_h_prev;

        %% Update Weights and Biases

        W_ho = W_ho + dW_ho;
        b_o = b_o + db_o;

        W_ih = W_ih + dW_ih;
        b_h = b_h + db_h;

        %% Store Previous Updates

        dW_ho_prev = dW_ho;
        db_o_prev = db_o;

        dW_ih_prev = dW_ih;
        db_h_prev = db_h;

    end

    %% Mean Squared Error

    MSE = totalError / size(X,1);
    MSE_history(epoch) = MSE;

    if mod(epoch,100) == 0
        fprintf('Epoch %d: MSE = %.6f\n', epoch, MSE);
    end

    %% Stop if Target MSE is Reached

    if MSE <= targetMSE

        fprintf('\nTraining converged at Epoch %d\n', epoch);
        fprintf('Final MSE = %.6f\n', MSE);

        break;

    end

end

%% Testing the Trained Network

fprintf('\nXOR Classification Results\n');
fprintf('---------------------------------------------\n');
fprintf('X1\tX2\tTarget\tNetwork Output\tBinary Output\n');
fprintf('---------------------------------------------\n');

for i = 1:size(X,1)

    % Forward propagation

    net_h = X(i,:) * W_ih + b_h;
    h = sigmoid(net_h);

    net_o = h * W_ho + b_o;
    y = sigmoid(net_o);

    % Binary output using threshold 0.5

    if y >= 0.5
        binaryOutput = 1;
    else
        binaryOutput = 0;
    end

    fprintf('%d\t%d\t%d\t%.4f\t\t%d\n', ...
        X(i,1), X(i,2), T(i), y, binaryOutput);

end

%% Plot Training Error

figure;

plot(1:epoch, MSE_history(1:epoch), 'LineWidth', 2);

xlabel('Epoch');
ylabel('Mean Squared Error');
title('XOR Backpropagation with Momentum');

grid on;