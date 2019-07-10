function [W, b] = InitializeParameters(m1, m2, sigma)
W = sigma*randn(m1, m2);
b = zeros(m1,1);
end