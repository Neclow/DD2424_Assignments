function accs = ComputeClassAccuracy(X_valid, Y_valid, y_valid, MFs, ConvNet)
%Validation: 14 examples per class
n = 14;
K = size(Y_valid, 1);

accs = zeros(1, K);
[~, ~, P] = FwdPass(X_valid, MFs, ConvNet.W);

n_start = 0;
for i = 0:K-1
    n_end = n_start + n;
    accs(i+1) = ComputeAccuracy(P(:,1+n_start:n_end), y_valid(1+n_start:n_end,:));
    n_start = n_end;
end
end