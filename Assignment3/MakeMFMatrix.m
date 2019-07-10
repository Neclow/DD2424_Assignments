function MF = MakeMFMatrix(F, n_len)
% Size of MF: (n_len-k+1)*nf x n_len*dd
[dd, k, nf] = size(F);

rows = (n_len-k+1)*nf;
cols = n_len*dd;
MF = zeros(rows, cols);

VF = [];
for i = 1:nf
    subF = F(:,:,i);
    f = subF(:)';
    VF = [VF; f];
end

for j = 0:n_len-k
    MF(j*nf+1:(j+1)*nf, 1+dd*j:size(VF,2)+dd*j) = VF;
end
end