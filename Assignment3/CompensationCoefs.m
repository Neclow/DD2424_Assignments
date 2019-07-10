function [n, p] = CompensationCoefs(X_train, Y_train)
K = size(Y_train, 1);
n = zeros(1, K);
p = zeros(1, K);
for i = 1:size(X_train,2)
    [~, y] = max(Y_train(:,i));
    n(y) = n(y) + 1;
end
for y = 1:K
    p(y) = 1/(n(y) * K);
end
end