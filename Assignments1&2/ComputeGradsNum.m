function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, dt)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

c = SVMCost(X, Y, W, b, lambda);

for j=1:length(b)
    grad_b = zeros(size(b));
    
    for i=1:length(b)
        b_try = b;
        b_try(i) = b_try(i) + dt;
        c2 = SVMCost(X, Y, W, b_try, lambda);
        grad_b(i) = (c2-c) / dt;
    end
end

for j=1:length(W)
    grad_W = zeros(size(W));
    
    for i=1:numel(W)   
        W_try = W;
        W_try(i) = W_try(i) + dt;
        c2 = SVMCost(X, Y, W_try, b, lambda);
        
        grad_W(i) = (c2-c) / dt;
    end
end