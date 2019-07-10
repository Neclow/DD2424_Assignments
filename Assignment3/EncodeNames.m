function vecName = EncodeNames(name, C, d, n_len)
%Y: one-hot-encoding of each character
matName = zeros(d, n_len);

%If test name has length greater than n_len, pick first n_len chars
if length(name) > n_len
    name = name(1:n_len);
end

for i = 1:length(name)
    [~, b] = find(C == name(i));
    matName(b,i) = 1;
end

vecName = matName(:);
end