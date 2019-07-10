function acc = ComputeAccuracy(P, y)
% P = FwdPass(X, W, b);
[~,k] = max(P);

acc = mean(k' == y);
end