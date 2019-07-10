function L = ComputeLoss(X, Y, RNN, h0)
tau = size(Y, 2);
L = 0;
h = h0;
for t = 1:tau
    xi = X(:,t);
    yi = Y(:,t);
    [~, h, pi] = FwdPass2(RNN, h, xi);
    L = L - log(yi'*pi);
end
end
