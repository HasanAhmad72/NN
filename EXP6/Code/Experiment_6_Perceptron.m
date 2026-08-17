clc;
clear;
close all;

%% Pattern Classification Using Perceptron Model

% Input patterns

% Class 1
X1 = [2 1;
      3 2;
      4 3;
      5 3];

% Class 2
X2 = [1 4;
      2 5;
      3 6;
      4 7];

% Combine input patterns
X = [X1; X2];

% Target outputs
% Class 1 = +1
% Class 2 = -1
T = [1; 1; 1; 1; -1; -1; -1; -1];

%% Add bias input
X_bias = [ones(size(X,1),1) X];

%% Initialize weights
rng(1); % For reproducible results
W = randn(1,3) * 0.1;

%% Learning parameters
learningRate = 0.1;
maxEpochs = 100;

%% Perceptron Training
for epoch = 1:maxEpochs

    errorCount = 0;

    for i = 1:size(X_bias,1)

        % Calculate net input
        net = W * X_bias(i,:)';

        % Perceptron activation function
        if net >= 0
            Y = 1;
        else
            Y = -1;
        end

        % Calculate error
        error = T(i) - Y;

        % Update weights if classification is incorrect
        if error ~= 0
            W = W + learningRate * error * X_bias(i,:);
            errorCount = errorCount + 1;
        end

    end

    fprintf('Epoch %d: Misclassified Patterns = %d\n', ...
        epoch, errorCount);

    % Stop training if all patterns are correctly classified
    if errorCount == 0
        fprintf('\nTraining converged at Epoch %d\n', epoch);
        break;
    end

end

%% Testing the trained perceptron
Y_pred = zeros(size(T));

for i = 1:size(X_bias,1)

    net = W * X_bias(i,:)';

    if net >= 0
        Y_pred(i) = 1;
    else
        Y_pred(i) = -1;
    end

end

%% Display Results
fprintf('\nPattern Classification Results\n');
fprintf('---------------------------------------------\n');
fprintf('X1\tX2\tTarget\tPredicted\tStatus\n');
fprintf('---------------------------------------------\n');

for i = 1:size(X,1)

    if T(i) == Y_pred(i)
        status = 'Correct';
    else
        status = 'Wrong';
    end

    fprintf('%d\t%d\t%d\t%d\t\t%s\n', ...
        X(i,1), X(i,2), T(i), Y_pred(i), status);

end

%% Calculate Accuracy
accuracy = sum(Y_pred == T) / length(T) * 100;

fprintf('\nClassification Accuracy = %.2f%%\n', accuracy);

%% Display Final Weights
fprintf('\nFinal Weights:\n');
fprintf('Bias Weight (W0) = %.4f\n', W(1));
fprintf('Weight W1 = %.4f\n', W(2));
fprintf('Weight W2 = %.4f\n', W(3));

%% Plot the patterns
figure;
hold on;
grid on;

% Plot Class 1
plot(X1(:,1), X1(:,2), 'ro', ...
    'MarkerSize', 8, 'LineWidth', 2);

% Plot Class 2
plot(X2(:,1), X2(:,2), 'bs', ...
    'MarkerSize', 8, 'LineWidth', 2);

%% Plot decision boundary
x1 = linspace(0,6,100);

% Equation:
% W0 + W1*x1 + W2*x2 = 0
x2 = -(W(1) + W(2)*x1) / W(3);

plot(x1, x2, 'k-', 'LineWidth', 2);

xlabel('Feature X1');
ylabel('Feature X2');
title('Pattern Classification Using Perceptron');

legend('Class 1', 'Class 2', 'Decision Boundary', ...
    'Location', 'best');

hold off;