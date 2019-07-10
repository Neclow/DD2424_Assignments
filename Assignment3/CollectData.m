function [L_train, L_valid, acc_train, acc_valid] = CollectData(X_train, Y_train, y_train, L_train, acc_train, X_valid, Y_valid, y_valid, L_valid, acc_valid, MFs, ConvNet, ustep, n_steps)
% Forward pass for loss/accuracy calculations
[~, ~, P_train] = FwdPass(X_train, MFs, ConvNet.W);
[~, ~, P_valid] = FwdPass(X_valid, MFs, ConvNet.W);

% Compute losses
L_train(ustep) = ComputeLoss(Y_train, P_train);
L_valid(ustep) = ComputeLoss(Y_valid, P_valid);

% Compute accuracies
acc_train(ustep) = ComputeAccuracy(P_train, y_train);
acc_valid(ustep) = ComputeAccuracy(P_valid, y_valid);

% Info on progress
disp(['Step: ', num2str(ustep), '/', num2str(n_steps)]);
disp(['L_train = ', num2str(L_train(ustep)), ', L_valid = ', num2str(L_valid(ustep))]);
disp(['acc_train = ', num2str(acc_train(ustep)), ', acc_valid = ', num2str(acc_valid(ustep))]);
end