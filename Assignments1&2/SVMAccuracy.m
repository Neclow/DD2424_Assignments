function acc = SVMAccuracy(X, y, W, b)
[~,k] = max(SVMClassifier(X, W, b));

acc = mean(k' == y);
end