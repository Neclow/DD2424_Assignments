function [X_batch1, X_batch2, P_batch] = FwdPass(X_batch, MFs, W)
% X_batch: (n_len*d) x n -> all vectorised input data
% Y_batch: one-hot encoding of labels of each ex
% W: 2D weight matrix for last fully connected layer

X_batch1 = max(MFs{1}*X_batch, 0);
X_batch2 = max(MFs{2}*X_batch1,0);

S_batch = W*X_batch2;

P_batch = softmax(S_batch);
end