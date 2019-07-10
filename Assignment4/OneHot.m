function Y = OneHot(ys, char_to_ind)
Y = zeros(length(char_to_ind), length(ys));
inds = zeros(1, length(ys));
for i = 1:length(ys)
    inds(i) = char_to_ind(ys(i));
    Y(inds(i), i) = 1; 
end
end