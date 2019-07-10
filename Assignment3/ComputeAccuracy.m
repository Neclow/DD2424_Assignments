function acc = ComputeAccuracy(P, y)
%P = EvaluateClassifier(X, W, b);
[~,k] = max(P);

acc = mean(k' == y);
end