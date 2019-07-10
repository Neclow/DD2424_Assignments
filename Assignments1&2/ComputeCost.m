function [L, J] = ComputeCost(P, Y, W, lambda)
% J scalar corresponding to the sum of the loss of the network's predictions
% for the images in X relative to the ground truth labels and regularization
% term on W
% P = FwdPass(X,W,b);

R = sumsqr(W);
L = -mean(sum(Y.*log(P)));
J = L + lambda*R;
end