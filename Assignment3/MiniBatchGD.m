function [ConvNet, ustep] = MiniBatchGD(X_train, Y_train, ConvNet, GDparams, n_len, n_len1, ustep)
n_batch = GDparams(1);
eta = GDparams(2);
gamma = GDparams(4);                                   % Momentum coef
N = size(X_train,2);

v_W = zeros(size(ConvNet.W));                          % Momentum of W
v_F = {zeros(size(ConvNet.F{1}), size(ConvNet.F{2}))}; % Momentum of F

MFs = cell(numel(ConvNet.F), 1);
MFs{1} = MakeMFMatrix(ConvNet.F{1}, n_len);
MFs{2} = MakeMFMatrix(ConvNet.F{2}, n_len1);

for j = 1:floor(N/n_batch)
    j_start = (j-1)*n_batch + 1;
    j_end = j*n_batch;
    inds = j_start:j_end;
    Xbatch = X_train(:, inds);
    Ybatch = Y_train(:, inds);
    [grad_F, grad_W] = ComputeGradients(Ybatch, Xbatch, ConvNet, MFs, n_len, n_len1);
    v_W = gamma*v_W + eta*grad_W;
    W = W - v_W;
    for k = 1:numel(ConvNet.F)    
        v_F{k} = gamma*v_F{k} + eta*grad_F{k};
        ConvNet.F{k} = ConvNet.F{k} - v_F{k};
    end
    ustep = ustep+1;                                   % Update step
end
end