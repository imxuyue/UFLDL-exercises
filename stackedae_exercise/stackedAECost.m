function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1)); % numClasses * M


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

% forward propagation to compute z, a
val = cell(numel(stack)+1, 1);%3

val{1}.a = data;
for d = 2:numel(val)
    val{d}.z = stack{d-1}.w*val{d-1}.a+repmat(stack{d-1}.b, 1, M);
    val{d}.a = sigmoid(val{d}.z);
end



% compute conditional probability
theta_x = softmaxTheta*val{numel(val)}.a; % numClasses * M
theta_x = bsxfun(@minus, theta_x, max(theta_x, [], 1)); % avoid overflow
e_theta_x = exp(theta_x);
p = bsxfun(@rdivide, e_theta_x, sum(e_theta_x));

% compute cost and softmaxThetaGrad
cost = (-1/M)*sum(sum(groundTruth.*log(p))) + (lambda/2)*sum(sum(softmaxTheta.^2));
softmaxThetaGrad = (-1/M)*((groundTruth-p)*val{numel(val)}.a') + lambda*softmaxTheta; % numClasses * N


% backpropagation
val{numel(val)}.delta = -softmaxTheta'*(groundTruth-p).*val{numel(val)}.a.*(1-val{numel(val)}.a);

for d = numel(val)-1:-1:2
    val{d}.delta = stack{d}.w'*val{d+1}.delta.*val{d}.a.*(1-val{d}.a);
end

for d = numel(stackgrad):-1:1
    stackgrad{d}.w = val{d+1}.delta*val{d}.a'/M;
    stackgrad{d}.b = sum(val{d+1}.delta, 2)/M;
end








% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
