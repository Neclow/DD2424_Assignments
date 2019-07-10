function X = Normalize(X, mean_X, std_X)
    X = X - repmat(mean_X, [1, size(X, 2)]);
    X = X ./ repmat(std_X, [1, size(X, 2)]);
end