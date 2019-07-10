function [ConvNet, L_train, L_valid, acc_train, acc_valid] = train(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, n_lens, ConvNet)
%n_epochs = GDparams(3);
ustep = 0; %Update step

n_len = n_lens(1);
n_len1 = n_lens(2);

% Loss, Accuracy data
L_train = zeros(n_epochs, 1);
L_valid = zeros(n_epochs, 1);
acc_train = zeros(n_epochs, 1);
acc_valid = zeros(n_epochs, 1);

% Filter matrix
MFs = {MakeMFMatrix(ConvNet.F{1}, n_len), MakeMFMatrix(ConvNet.F{2}, n_len1)};

%Evaluate classifier for loss/cost/accuracy calculations
[~, ~, P_train] = FwdPass(X_train, MFs, ConvNet.W);
[~, ~, P_valid] = FwdPass(X_valid, MFs, ConvNet.W);

%Initial loss/cost/accuracy values (before training)
L_train(1) = ComputeLoss(X_train, Y_train, MFs, ConvNet);
L_valid(1) = ComputeLoss(X_valid, Y_valid, MFs, ConvNet);
acc_train(1) = ComputeAccuracy(P_train, y_train);
acc_valid(1) = ComputeAccuracy(P_valid, y_valid);

for t = 1:n_epochs
    [ConvNet, ustep] = MiniBatchGD(X_train, Y_train, ConvNet, GDparams, n_len, n_len1, ustep);
    
     %Evaluate classifier for loss/cost/accuracy calculations
    [~, ~, P_train] = FwdPass(X_train, MFs, Wstar);
    [~, ~, P_valid] = FwdPass(X_valid, MFs, Wstar);
    
    %Compute loss and costs
    L_train(t+1) = ComputeLoss(P_train, Y_train, Wstar, lambda);
    L_valid(t+1) = ComputeLoss(P_valid, Y_valid, Wstar, lambda);
    
    %Compute accuracies
    acc_train(t+1) = ComputeAccuracy(P_train, y_train);
    acc_valid(t+1) = ComputeAccuracy(P_valid, y_valid);
    
    %Info on progress
    disp(['Step ', num2str(t), '/', num2str(n_epochs)]);
    disp(['J_train = ', num2str(J_train(t+1)), ', J_valid = ', num2str(J_valid(t+1))]);
    disp(['acc_train = ', num2str(acc_train(t+1)), ', acc_valid = ', num2str(acc_valid(t+1))]);
end
end