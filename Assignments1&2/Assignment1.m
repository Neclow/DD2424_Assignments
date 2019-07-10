%% Assignment 1
function Assignment1
rng(400);
disp('Bonuses available:')
disp('1 = train with all data')
disp('2 = training until overfit')
disp('3 = train with decaying LR')
disp('4 = train with SVM multi-class loss')
disp('other = default training')
bonus = input('Enter desired bonus: ');

switch bonus
    case 1
        disp('You selected: training with all data')
    case 2
        disp('You selected: training until overfit')
    case 3
        disp('You selected: train with decaying LR')
    case 4
        disp('You selected: train with SVM multi-class loss')
    otherwise
        disp('You selected: default training')
end

if bonus == 1
    [X1, Y1, y1] = LoadBatch('Datasets\cifar-10-batches-mat\data_batch_1.mat'); % Training
    [X2, Y2, y2] = LoadBatch('Datasets\cifar-10-batches-mat\data_batch_2.mat'); % Training
    [X3, Y3, y3] = LoadBatch('Datasets\cifar-10-batches-mat\data_batch_3.mat'); % Training
    [X4, Y4, y4] = LoadBatch('Datasets\cifar-10-batches-mat\data_batch_4.mat'); % Training
    [X5, Y5, y5] = LoadBatch('Datasets\cifar-10-batches-mat\data_batch_5.mat'); % Training
    X = [X1, X2, X3, X4, X5];
    Y = [Y1, Y2, Y3, Y4, Y5];
    y = [y1; y2; y3; y4; y5];
    valid_inds = sort(randperm(length(X), 1000)); % Select 1000 random indices for validation set
    all_inds = 1:length(X);
    common_inds = intersect(all_inds, valid_inds);
    train_inds = setxor(all_inds, common_inds);   % Other indices become part of the training set

    X_valid = PartitionData(X, valid_inds);
    Y_valid = PartitionData(Y, valid_inds);
    y_valid = PartitionData(y', valid_inds);
    y_valid = y_valid';
    X_train = PartitionData(X, train_inds);
    Y_train = PartitionData(Y, train_inds);
    y_train = PartitionData(y', train_inds);
    y_train = y_train';
else
    [X_train, Y_train, y_train] = LoadBatch('Datasets\cifar-10-batches-mat\data_batch_1.mat'); %Training
    [X_valid, Y_valid, y_valid] = LoadBatch('Datasets\cifar-10-batches-mat\data_batch_2.mat'); %Validation    
end

[X_test, ~, y_test] = LoadBatch('Datasets\cifar-10-batches-mat\test_batch.mat'); % Test

% Initialize W, b
K = 10;
d = 3072; % 32*32*3
sigma = 0.01; % SD of Gaussian deviation
W = sigma*randn(K, d);
b = sigma*randn(K, 1);

% Hyperparameters
GDparams.n_batch = 100;
GDparams.eta = input('Enter LR [0.1, 0.01]: ');
GDparams.n_epochs = 40;
if bonus == 3
    GDparams.w_decay = 0.9;
else 
    GDparams.w_decay = 1;
end
GDparams.lambda = input('Enter regularization factor [0, 0.1, 1]: ');

tic
disp('Training...')
[bstar, Wstar, J_train, J_valid, L_train, L_valid, acc_train, acc_valid] = train(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, GDparams, W, b, bonus);

% Final test accuracy
if bonus == 4
   acc_test = SVMAccuracy(X_test, y_test, Wstar, bstar);
else
   acc_test = ComputeAccuracy(X_test, y_test, Wstar, bstar); 
end
disp(['Validation accuracy: ', num2str(acc_valid(end)*100), '%'])
disp(['Test accuracy: ', num2str(acc_test*100), '%'])
toc

%% Plotting
t = 1:size(J_train, 1);

figure(1)
plot(t, J_train', 'k--o', 'DisplayName', 'training');
hold on
plot(t, J_valid', 'r--o', 'DisplayName', 'validation');
xlabel('epoch');
ylabel('Cost');
legend show
grid on

figure(2)
plot(t, L_train', 'k--o', 'DisplayName', 'training');
hold on
plot(t, L_valid', 'r--o', 'DisplayName', 'validation');
xlabel('epoch');
ylabel('Loss');
legend show
grid on

figure(3)
plot(t, acc_train', 'k--o', 'DisplayName', 'training');
hold on
plot(t, acc_valid', 'r--o', 'DisplayName', 'validation');
xlabel('epoch');
ylabel('Accuracy');
legend show
grid on

% % Generate predicted images
figure(4)
L=ceil(10^.5);
for i=1:10
    im = reshape(Wstar(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
    subplot(L,L,i);
    imshow(s_im{i});
end
end

%% Reading data
function [X, Y, y] = LoadBatch(filename)
A = load(filename);
X = double(A.data')/255;
y = double(A.labels)+ones(1e4,1);
Y = eye(max(y));
Y = Y(y,:)';
% Reads data from CIFAR-10 batch file
% Out: image and label data in separate files
% X: image pixel data, size d x N
% Y: One-hot representation of the label for each image, size K x N
% y: label for each image, size N x 1
% N = nb of images (10000), d = dimensionality of each image (32x32x3)
% K = nb of labels (10)
end

% Select subdata Xp from data X according to generated random indices
function Xp = PartitionData(X, inds)
% X: original dataset
% inds: indices selected for partition dataset
% Xp: partitioned dataset

Xp = zeros(size(X,1), length(inds));
for i = 1:length(inds)
    Xp(:,i) = X(:,inds(i));
end
end

%% Evaluation of network function
function P = EvaluateClassifier(X, W, b)
% Each column of X corresponds to an image, size d x n
% W, size K x d
% P contains proba for each label for the image in the corresponding X
% size K x n
S = W*X + b;
P = softmax(S);
end

function S = SVMClassifier(X, W, b)
S = W*X + b;
end

%% Cost and loss computation
function [L, J] = ComputeCost(X, Y, W, b, lambda)
% Each column of X corresponds to an image, size d x n
% Each column of Y = one-hot ground truth label for the corresponding column
% of X
% J scalar corresponding to the sum of the loss of the network's predictions
% for the images in X relative to the ground truth labels and regularization
% term on W
l = -sum(Y.*log(EvaluateClassifier(X,W,b)));
L = mean(l);
R = sumsqr(W);
J = L + lambda*R;
end

% SVM cost (for calculation of validation loss/cost)
function [L, J] = SVMCost(X, Y, W, b, lambda)
N = size(X, 2); % Nb of imgs
C = size(Y, 1); % Nb of classes

L = 0;

for i = 1:N
    xi = X(:,i);
    yi = Y(:,i);
    si = SVMClassifier(xi, W, b);
    [~, ki] = max(yi);
    sy = si(ki);
    for j = 1:C
        if j == ki
            continue
        end
        lj = si(j) - sy + 1;
        if lj > 0
            L = L + lj;
        end
    end
end

L = L/N;

J = L + 0.5*lambda*sumsqr(W);
end

%% Accuracy computations
function acc = ComputeAccuracy(X,y,W,b)
[~,k] = max(EvaluateClassifier(X, W, b));

acc = mean(k' == y);
end

function acc = SVMAccuracy(X, y, W, b)
[~,k] = max(SVMClassifier(X, W, b));

acc = mean(k' == y);
end

%% Gradient computations
function [grad_b, grad_W] = ComputeGradients(X, Y, P, W, lambda)
%Maximum error was found to be 2.3627e-07 compared with ComputeGradsNum.m
%(with 1-dimension X_train, Y_train, P_train).
N = size(X,2);

G = -(Y-P);

grad_W = (G*X')./N + 2*lambda*W;
grad_b = G*ones(1,N)'./N;
end

function [L, J, grad_b, grad_W] = SVMPass(X, Y, W, b, lambda)
% L, J: loss/cost from FORWARD pass
% grad_b, grad_W: gradients from BACKWARD PASS
grad_W = zeros(size(W));
grad_b = zeros(size(b));

N = size(X, 2); % Nb of imgs
C = size(Y, 1); % Nb of classes

L = 0;

for i = 1:N
    xi = X(:,i);
    yi = Y(:,i);
    si = SVMClassifier(xi, W, b); % Evaluate classifier for image i
    [~, ki] = max(yi);            % Label for image i
    sy = si(ki);                  % Corresponding score for label i
    for j = 1:C
        if j == ki
            continue
        end
        lj = si(j) - sy + 1;
        if lj > 0
            % Loss calculations (fwd pass)
            L = L + lj;
            
            % Gradient calcuations (bwd pass)
            grad_W(ki,:) = grad_W(ki,:) - xi';
            grad_W(j,:) = grad_W(j,:) + xi';
            grad_b(ki) = grad_b(ki) - 1;
            grad_b(j) = grad_b(j) + 1;
        end
    end
end

% Averaging loss and gradients
L = L/N;
grad_W = grad_W/N;
grad_b = grad_b/N;

% Adding regularization
J = L + 0.5*lambda*sumsqr(W);
grad_W = grad_W + lambda*W;
end

%% SGD w/ mini-batch
function [bstar, Wstar] = MiniBatchGD(X_train, Y_train, GDparams, Wstar, bstar)
N = size(X_train,2);
for j = 1:N/GDparams.n_batch
    j_start = (j-1)*GDparams.n_batch + 1;
    j_end = j*GDparams.n_batch;
    inds = j_start:j_end;
    Xbatch = X_train(:, inds);
    Ybatch = Y_train(:, inds);
    
    % Fwd pass
    Pbatch = EvaluateClassifier(Xbatch, Wstar, bstar);
    
    % Bwd pass
    [grad_b, grad_W] = ComputeGradients(Xbatch, Ybatch, Pbatch, Wstar, GDparams.lambda);
    Wstar = Wstar - GDparams.eta*grad_W;
    bstar = bstar - GDparams.eta*grad_b;
 end
end

% Mini-batch gradient integrating SVM fwd/bwd passes
function [L, J, bstar, Wstar] = SVMGD(X_train, Y_train, GDparams, Wstar, bstar)
N = size(X_train,2);
for j = 1:N/GDparams.n_batch
    j_start = (j-1)*GDparams.n_batch + 1;
    j_end = j*GDparams.n_batch;
    inds = j_start:j_end;
    Xbatch = X_train(:, inds);
    Ybatch = Y_train(:, inds);
    
    % Fwd + bwd pass
    [L, J, grad_b, grad_W] = SVMPass(Xbatch, Ybatch, Wstar, bstar, GDparams.lambda);
    
    % Update W, b
    Wstar = Wstar - GDparams.eta*grad_W;
    bstar = bstar - GDparams.eta*grad_b;
 end
end

%% Overall training
function [bstar, Wstar, J_train, J_valid, L_train, L_valid, acc_train, acc_valid] = train(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, GDparams, W, b, bonus)
bstar = b;
Wstar = W;

J_train = zeros(GDparams.n_epochs, 1);
L_train = zeros(GDparams.n_epochs, 1);
J_valid = zeros(GDparams.n_epochs, 1);
L_valid = zeros(GDparams.n_epochs, 1);
acc_train = zeros(GDparams.n_epochs, 1);
acc_valid = zeros(GDparams.n_epochs, 1);

if bonus == 2 % Bonus 2: train until overfit
    % Initial loss/cost
    t = 1;
    [L_train(t), J_train(t)] = ComputeCost(X_train, Y_train, Wstar, bstar, GDparams.lambda);
    [L_valid(t), J_valid(t)] = ComputeCost(X_valid, Y_valid, Wstar, bstar, GDparams.lambda);
    loss_valid = L_valid(end); % Validation accuracy for epoch n-1
    
    while J_valid(end) <= loss_valid
        t = t+1;
        
        [bstar, Wstar] = MiniBatchGD(X_train, Y_train, GDparams, W, b);
        
        [L_train(t), J_train(t)] = ComputeCost(X_train, Y_train, Wstar, bstar, GDparams.lambda);
        [L_valid(t), J_valid(t)] = ComputeCost(X_valid, Y_valid, Wstar, bstar, GDparams.lambda);
        loss_valid = L_valid(t-1);
        
        if mod(t, 50) == 0
            disp(['Epoch: ', num2str(t)])
            disp(['L_valid(end) = ', num2str(L_valid(end)), ', L_valid(end-1) = ', num2str(acc_valid)])
        end
        
        acc_train(t) = ComputeAccuracy(X_train, y_train, Wstar, bstar);
        acc_valid(t) = ComputeAccuracy(X_valid, y_valid, Wstar, bstar);
    end
elseif bonus == 4 % Bonus 4: train with SVM loss
    for t = 1:GDparams.n_epochs
        [L, J, bstar, Wstar] = SVMGD(X_train, Y_train, GDparams, Wstar, bstar);
        
        L_train(t) = L;
        J_train(t) = J;
        [L_valid(t), J_valid(t)] = SVMCost(X_valid, Y_valid, Wstar, bstar, GDparams.lambda);
        
        acc_train(t) = SVMAccuracy(X_train, y_train, Wstar, bstar);
        acc_valid(t) = SVMAccuracy(X_valid, y_valid, Wstar, bstar);
        
        if mod(t, 10) == 0
            disp(['Epoch ', num2str(t), '/', num2str(GDparams.n_epochs)])
        end
    end
else % Default training
    for t = 1:GDparams.n_epochs
        [bstar, Wstar] = MiniBatchGD(X_train, Y_train, GDparams, Wstar, bstar);
        
        GDparams.eta = GDparams.eta*GDparams.w_decay; % Decay LR (if weight_decay < 1)
        
        [L_train(t), J_train(t)] = ComputeCost(X_train, Y_train, Wstar, bstar, GDparams.lambda);
        [L_valid(t), J_valid(t)] = ComputeCost(X_valid, Y_valid, Wstar, bstar, GDparams.lambda);
        
        acc_train(t) = ComputeAccuracy(X_train, y_train, Wstar, bstar);
        acc_valid(t) = ComputeAccuracy(X_valid, y_valid, Wstar, bstar);
        
        if mod(t, 10) == 0
            disp(['Epoch ', num2str(t), '/', num2str(GDparams.n_epochs)])
        end
    end
end
disp('Training complete.')
if bonus == 2
    disp(['Number of epochs before overfit: ', num2str(t)])
elseif bonus == 3 % Check for weight decay
    disp(['Final LR: ', num2str(GDparams.eta)])
end
end