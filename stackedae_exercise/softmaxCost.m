function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1)); % numClasses * M
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

[N, M] = size(data);

theta_x = theta*data; % numClasses * M
theta_x = bsxfun(@minus, theta_x, max(theta_x, [], 1)); % avoid overflow
e_theta_x = exp(theta_x);
p = bsxfun(@rdivide, e_theta_x, sum(e_theta_x));

cost = (-1/M)*sum(sum(groundTruth.*log(p))) + (lambda/2)*sum(sum(theta.^2));

thetagrad = (-1/M)*((groundTruth-p)*data') + lambda*theta; % numClasses * N


% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

