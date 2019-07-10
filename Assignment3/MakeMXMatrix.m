function MX = MakeMXMatrix(x_input, d, k, nf, n_len)
X_input = reshape(x_input, [d, n_len]);
rows = (n_len-k+1)*nf;
cols = (k*d)*nf;

MX = zeros(rows, cols);

for i = 0:n_len-k
    subMX = zeros(nf,cols);
    for j = 1:nf
         subX = X_input(:, i+1:i+k);        
         subx = subX(:)';
         subMX(j, (j-1)*length(subx)+1:j*length(subx)) = subx;
    end
    MX(i*nf+1:(i+1)*nf,:) = subMX;
end
end