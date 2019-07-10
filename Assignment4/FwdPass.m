function [a, h, p] = FwdPass(RNN, h0, X)
[N, tau] = size(X);

% Build a, h, p vectors
l = length(h0);
a = zeros(l, tau);
h = zeros(l, tau+1);
p = zeros(N, tau);
h(:,1) = h0;

for t = 1:tau
    xt = X(:, t);
    ht = h(:,t);
    [a(:,t), h(:,t+1), p(:,t)] = FwdPass2(RNN, ht, xt);
end
end