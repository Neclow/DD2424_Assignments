function [L, J] = SVMCost(X, Y, W, b, lambda)
N = size(X, 2); % Nb of imgs
C = size(Y, 1); % Nb of classes

L = 0;

for i = 1:N
    xi = X(:,i);
    yi = Y(:,i);
    si = SVMClassifier(xi, W, b);
    [~, ki] = max(yi);
    sy = si(ki);
    for j = 1:C
        if j == ki
            continue
        end
        lj = si(j) - sy + 1;
        if lj > 0
            L = L + lj;
        end
    end
end

L = L/N;

J = L + 0.5*lambda*sumsqr(W);
end