function L = ComputeLoss(Ys_batch, P_batch)
%[~, ~, P_batch] = FwdPass(X_batch, MFs, ConvNet.W);

% if comp
%     L = -mean(sum(p*(Ys_batch.*log(P_batch))));
% else 
%     L = -mean(sum(Ys_batch.*log(P_batch)));
% end

L = -mean(sum(Ys_batch.*log(P_batch)));
% R = sumsqr(W{1}) + sumsqr(W{2});
% J = L + lambda*R;
end