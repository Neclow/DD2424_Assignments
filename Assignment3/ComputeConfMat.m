function C = ComputeConfMat(X_valid, y_valid, MFs, ConvNet)
[~, ~, P_valid] = FwdPass(X_valid, MFs, ConvNet.W);
[~, k] = max(P_valid);

C = confusionmat(k, y_valid);
end