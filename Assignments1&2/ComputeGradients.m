function [grad_b, grad_W] = ComputeGradients(X, Y, W, b, lambda)
N = size(X,2);

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

for i = 1:numel(W)
    grad_W{i} = zeros(size(W{i}));
    grad_b{i} = zeros(size(b{i}));    
end

[P, H, ~, ~] = FwdPassInvDropout(X, W, b, 0.5);

% Efficient computations for mini-batch gradient of loss, slide 30
G = -(Y-P);

for i = flip(2:numel(W))
    grad_W{i} = G*H{i-1}';
    grad_b{i} = G*ones(N,1);
    G = W{i}'*G;
    G = G.*(H{i-1} > 0);
end

grad_W{1} = G*X';
grad_b{1} = G*ones(N,1);

%Divide by nb of entries and add regularization term
for j = 1:length(W)
    grad_W{j} =  grad_W{j}/N + 2*lambda*W{j};
    grad_b{j} =  grad_b{j}/N;
end
end