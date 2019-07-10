function [bstar, Wstar, L_train, L_valid, J_train, J_valid, acc_train, acc_valid] = train(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, GDparams, W, b, lambda)
n_epochs = GDparams(3);
ustep = 0; % Update step

% Loss, Cost, Accuracy data
L_train = zeros(n_epochs, 1);
L_valid = zeros(n_epochs, 1);
J_train = zeros(n_epochs, 1);
J_valid = zeros(n_epochs, 1);
acc_train = zeros(n_epochs, 1);
acc_valid = zeros(n_epochs, 1);

Wstar = W;
bstar = b;

% Evaluate classifier for loss/cost/accuracy calculations
P_train = FwdPass(X_train, Wstar, bstar);
P_valid = FwdPass(X_valid, Wstar, bstar);

% Initial loss/cost/accuracy values (before training)
[L_train(1), J_train(1)] = ComputeCost(P_train, Y_train, Wstar, lambda);
[L_valid(1), J_valid(1)] = ComputeCost(P_valid, Y_valid, Wstar, lambda);
acc_train(1) = ComputeAccuracy(P_train, y_train);
acc_valid(1) = ComputeAccuracy(P_valid, y_valid);

for t = 1:n_epochs
    [bstar, Wstar, GDparams, ustep] = MiniBatchGD(X_train, Y_train, GDparams, Wstar, bstar, lambda, ustep);
    
    % Evaluate classifier for loss/cost/accuracy calculations
    P_train = FwdPass(X_train, Wstar, bstar);
    P_valid = FwdPass(X_valid, Wstar, bstar);
    
    % Compute loss and costs
    [L_train(t+1), J_train(t+1)] = ComputeCost(P_train, Y_train, Wstar, lambda);
    [L_valid(t+1), J_valid(t+1)] = ComputeCost(P_valid, Y_valid, Wstar, lambda);
    
    %Compute accuracies
    acc_train(t+1) = ComputeAccuracy(P_train, y_train);
    acc_valid(t+1) = ComputeAccuracy(P_valid, y_valid);
    
    %Info on progress
    disp(['Step ', num2str(t), '/', num2str(n_epochs)]);
    %disp(['J_train = ', num2str(J_train(t+1)), ', J_valid = ', num2str(J_valid(t+1))]);
    disp(['acc_train = ', num2str(acc_train(t+1)), ', acc_valid = ', num2str(acc_valid(t+1))]);
end
end