function [bstar, Wstar] = SVMGD(X_train, GDparams, Wstar, bstar)
N = size(X_train,2);
for j = 1:N/GDparams.n_batch
    j_start = (j-1)*GDparams.n_batch + 1;
    j_end = j*GDparams.n_batch;
    inds = j_start:j_end;
    Xbatch = X_train(:, inds);
    
    % Bwd pass
    [grad_b, grad_W] = SVMGradients(Xbatch, Wstar, bstar, GDparams.lambda);
    Wstar = Wstar - GDparams.eta*grad_W;
    bstar = bstar - GDparams.eta*grad_b;
 end
end