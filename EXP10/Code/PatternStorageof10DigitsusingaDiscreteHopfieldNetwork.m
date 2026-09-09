clc;
clear;
close all;

%% Pattern Storage of 10 Digits using Discrete Hopfield Network
% 7 x 5 patterns for digits 0 to 9

%% Define digit patterns

D{1} = [
    0 1 1 1 0;
    1 0 0 0 1;
    1 0 0 1 1;
    1 0 1 0 1;
    1 1 0 0 1;
    1 0 0 0 1;
    0 1 1 1 0
];

D{2} = [
    0 0 1 0 0;
    0 1 1 0 0;
    0 0 1 0 0;
    0 0 1 0 0;
    0 0 1 0 0;
    0 0 1 0 0;
    0 1 1 1 0
];

D{3} = [
    0 1 1 1 0;
    1 0 0 0 1;
    0 0 0 0 1;
    0 0 1 1 0;
    0 1 0 0 0;
    1 0 0 0 0;
    1 1 1 1 1
];

D{4} = [
    1 1 1 1 0;
    0 0 0 0 1;
    0 0 0 1 0;
    0 0 1 1 0;
    0 1 0 1 0;
    1 1 1 1 1;
    0 0 0 1 0
];

D{5} = [
    1 1 1 1 1;
    1 0 0 0 0;
    1 0 0 0 0;
    1 1 1 1 0;
    0 0 0 0 1;
    0 0 0 0 1;
    1 1 1 1 0
];

D{6} = [
    0 1 1 1 0;
    1 0 0 0 0;
    1 0 0 0 0;
    1 1 1 1 0;
    1 0 0 0 1;
    1 0 0 0 1;
    0 1 1 1 0
];

D{7} = [
    1 1 1 1 1;
    0 0 0 0 1;
    0 0 0 1 0;
    0 0 1 0 0;
    0 1 0 0 0;
    0 1 0 0 0;
    0 1 0 0 0
];

D{8} = [
    0 1 1 1 0;
    1 0 0 0 1;
    1 0 0 0 1;
    0 1 1 1 0;
    1 0 0 0 1;
    1 0 0 0 1;
    0 1 1 1 0
];

D{9} = [
    0 1 1 1 0;
    1 0 0 0 1;
    1 0 0 0 1;
    0 1 1 1 1;
    0 0 0 0 1;
    0 0 0 0 1;
    0 1 1 1 0
];

D{10} = [
    0 1 1 1 0;
    1 0 0 0 1;
    1 0 0 0 1;
    0 1 1 1 1;
    0 0 0 0 1;
    0 0 0 0 1;
    0 1 1 1 0
];

%% Convert patterns to bipolar form
P = zeros(35,10);

for k = 1:10
    temp = D{k};
    temp(temp == 0) = -1;
    P(:,k) = temp(:);
end

%% Calculate Hopfield weight matrix
W = P * P';

% Remove self-connections
W(1:36:end) = 0;

%% Test digit 8
testDigit = 8;
x = P(:,testDigit);

%% Add noise
noiseLevel = 0.15;

rng(1);

noise = rand(size(x)) < noiseLevel;

xNoisy = x;
xNoisy(noise) = -xNoisy(noise);

%% Recall using Hopfield Network
xRecall = xNoisy;

for iteration = 1:20

    oldState = xRecall;

    for i = 1:35

        net = W(i,:) * xRecall;

        if net >= 0
            xRecall(i) = 1;
        else
            xRecall(i) = -1;
        end

    end

    if isequal(xRecall,oldState)
        break;
    end

end

%% Identify recalled digit
similarity = P' * xRecall;

[~,index] = max(similarity);

recalledDigit = index - 1;

fprintf('Original Digit = %d\n',testDigit);
fprintf('Recalled Digit = %d\n',recalledDigit);
fprintf('Number of Iterations = %d\n',iteration);

%% Display results
figure;

subplot(1,3,1);
imagesc(reshape(x,7,5));
axis image;
axis off;
title('Original Digit');

subplot(1,3,2);
imagesc(reshape(xNoisy,7,5));
axis image;
axis off;
title('Noisy Digit');

subplot(1,3,3);
imagesc(reshape(xRecall,7,5));
axis image;
axis off;
title('Recalled Digit');

colormap(gray);