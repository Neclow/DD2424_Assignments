function [grad_F, grad_W] = ComputeGradients(Y_batch, X_batch, ConvNet, MFs, MX1, n_lens)
%n_len = n_lens(1);
n_len1 = n_lens(2);
n_len2 = n_lens(3);

[X_batch1, X_batch2, P_batch] = FwdPass(X_batch, MFs, ConvNet.W);

grad_F = {zeros(1, length(ConvNet.F{1}(:))), zeros(1, length(ConvNet.F{2}(:)))};

[n1, k2, n2] = size(ConvNet.F{2});

G_batch = -(Y_batch-P_batch);

N = size(X_batch, 2);

%grad_W = (1/N)*p.*G_batch*X_batch2';
grad_W = (1/N)*G_batch*X_batch2';

G_batch = ConvNet.W'*G_batch;
G_batch = G_batch.*(X_batch2 > 0);

for j = 1:N
    gj = G_batch(:,j);
    Gj = reshape(gj, [n2 n_len2]).';
    xj = X_batch1(:,j);
    V = sparse(MakeMXMatrix2(xj, n1, k2, n_len1))'*Gj;
    grad_F{2} = grad_F{2} + V(:)'/N;
end

G_batch = MFs{2}'*G_batch;
G_batch = G_batch.*(X_batch1 > 0);

for j = 1:N
    gj = G_batch(:,j);
    v = gj'*MX1{1,j};
    grad_F{1} = grad_F{1} + v/N;
end

grad_F{1} = reshape(grad_F{1}, size(ConvNet.F{1}));
grad_F{2} = reshape(grad_F{2}, size(ConvNet.F{2}));
end