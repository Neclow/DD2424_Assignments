function [b, W, GDparams, ustep] = MiniBatchGD(X_train, Y_train, GDparams, W, b, lambda, ustep)
n_batch = GDparams(1);
eta = GDparams(2);
n_s = GDparams(4);
eta_min = GDparams(5);
eta_max = GDparams(6);
%gamma = GDparams(7);
N = size(X_train,2);

%Momentum of W
% v_W = cell(numel(W), 1);
% v_W{1} = zeros(size(W{1}));
% v_W{2} = zeros(size(W{2}));

%Momentum of b
% v_b = cell(numel(b), 1);
% v_b{1} = zeros(size(b{1}));
% v_b{2} = zeros(size(b{2}));

for j = 1:N/n_batch
    eta = UpdateLearningRate(eta, ustep, n_s, eta_min, eta_max);
    j_start = (j-1)*n_batch + 1;
    j_end = j*n_batch;
    inds = j_start:j_end;
    Xbatch = X_train(:, inds);
    Ybatch = Y_train(:, inds);
    [grad_b, grad_W] = ComputeGradients(Xbatch, Ybatch, W, b, lambda);
    for k = 1:numel(W)
        W{k} = W{k} - eta*grad_W{k};
        b{k} = b{k} - eta*grad_b{k};
%         v_W{k} = gamma*v_W{k} + eta*grad_W{k};       
%         v_b{k} = gamma*v_b{k} + eta*grad_b{k};
%         W{k} = W{k} - v_W{k};
%         b{k} = b{k} - v_b{k};
    end
    ustep = ustep+1; %Update step
end

GDparams(2) = eta;
end