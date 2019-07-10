function MX = MakeMXMatrix2(x_input, d, k, n_len)
X_input = reshape(x_input, [d, n_len]);
rows = (n_len-k+1);
cols = k*d;

MX = zeros(rows, cols);
for i = 1:n_len-k+1
    subX = X_input(:, i:i+k-1);
    subx = subX(:)';
    MX(i, :) = subx;
end
end