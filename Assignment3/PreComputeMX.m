function MX1s = PreComputeMX(X, d1, k1, n1, n_len)
disp('Pre-computing MX...');
tic
N = size(X, 2);
MX1s = cell(1,N);
for j = 1:N
    xj = X(:,j); 
    MX1s{1,j} = sparse(MakeMXMatrix(xj, d1, k1, n1, n_len));
end
toc
end