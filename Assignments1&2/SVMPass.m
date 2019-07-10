function [L, J, grad_b, grad_W] = SVMPass(X, Y, W, b, lambda)
grad_W = zeros(size(W));
grad_b = zeros(size(b));

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
            grad_W(ki,:) = grad_W(ki,:) - xi';
            grad_W(j,:) = grad_W(j,:) + xi';
            grad_b(ki) = grad_b(ki) - 1;
            grad_b(j) = grad_b(j) + 1;
        end
    end
end

L = L/N;
grad_W = grad_W/N;
grad_b = grad_b/N;

J = L + 0.5*lambda*sumsqr(W);
grad_W = grad_W + lambda*W;
end