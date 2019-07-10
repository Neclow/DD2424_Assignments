%% Initialization
disp('Loading data...')
[X1, Y1, y1] = LoadBatch('Datasets\cifar-10-batches-mat\data_batch_1.mat');
[X2, Y2, y2] = LoadBatch('Datasets\cifar-10-batches-mat\data_batch_2.mat'); 
[X3, Y3, y3] = LoadBatch('Datasets\cifar-10-batches-mat\data_batch_3.mat'); 
[X4, Y4, y4] = LoadBatch('Datasets\cifar-10-batches-mat\data_batch_4.mat');
[X5, Y5, y5] = LoadBatch('Datasets\cifar-10-batches-mat\data_batch_5.mat'); 
[X_test, Y_test, y_test] = LoadBatch('Datasets\cifar-10-batches-mat\test_batch.mat'); %Test

disp('Done')
disp('Preparing data...')

X = [X1, X2, X3, X4, X5];
Y = [Y1, Y2, Y3, Y4, Y5];
y = [y1; y2; y3; y4; y5];
N = size(X, 2);

%Training data
n_valid = 5000;
X_train = X(:,1:(N-n_valid));
Y_train = Y(:,1:(N-n_valid));
y_train = y(1:(N-n_valid),:);

%Validation data
X_valid = X(:,(N-n_valid+1):N);
Y_valid = Y(:,(N-n_valid+1):N);
y_valid = y((N-n_valid+1):N,:);

mean_X = mean(X_train, 2);
std_X = std(X_train, 0, 2);
X_train = Normalize(X_train, mean_X, std_X);
X_valid = Normalize(X_valid, mean_X, std_X);
X_test = Normalize(X_test, mean_X, std_X);
disp('Done')

% Initialize W, b
rng(400);
d = size(X_train, 1);
K = size(Y_train, 1);
N = size(X_train, 2);
m = [d, 50, K]; % Generalization to k-layer NNs
sigma = zeros(length(m)-1, 1);
W = cell(length(m)-1, 1);
b = cell(length(m)-1, 1);
for i = 1:length(m)-1
    sigma(i) = 1/sqrt(m(i));
    [W{i}, b{i}] = InitializeParameters(m(i+1), m(i), sigma(i));
end

n_batch = 100;
eta = 0.1;
%n_s = 2*floor(N/n_batch);
%n_cycles = 2;
% n_epochs = 2*n_cycles*n_s/(N/n_batch);
eta_min = 1e-5;
eta_max = 1e-1;
gamma = 0.9; % Momentum coefficient
p = 0.5; % Dropout coefficient
% GDparams = [n_batch, eta, n_epochs, n_s, eta_min, eta_max, gamma, p];
% lambda = 2.574355e-4; % Optimal LR, 50 nodes
% lambda = 0.003576; % Optimal LR, 100 nodes, no dropout
% lambda = 0.002362; % Optimal LR, 200 nodes, no dropout
% lambda = 2.134e-05; % Optimal LR, 100 nodes, dropout = 0.5
% lambda = 5.708e-05; % Optimal LR, 200 nodes, dropout = 0.5

%% Gradient computations
% W{1} = W{1}(:,1:100);
% 
% [grad_b, grad_W] = ComputeGradients(X_train(1:100, 1:n_batch), Y_train(:,1:n_batch), W, b, lambda);
% [ngrad_b, ngrad_W] = ComputeGradsNumSlow(X_train(1:100, 1:n_batch), Y_train(:,1:n_batch), W, b, lambda, 1e-5);
% [err_b, err_W] = ComputeRelativeError(ngrad_b, grad_b, ngrad_W, grad_W);

%% Training with cyclical rates
% [bstar, Wstar, L_train, L_valid, J_train, J_valid, acc_train, acc_valid] = train(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, GDparams, W, b, lambda);
% 
% % Test accuracy
% P_test = FwdPass(X_test, Wstar, bstar);
% acc_test = ComputeAccuracy(P_test, y_test);
% 
% PlotResults(J_train, J_valid, acc_train, acc_valid, L_train, L_valid, n_epochs);

%% Coarse-to-fine random search of lambda
% lambda_data = SearchLambdaOpt(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, GDparams, W, b, -5, -1);
% lambda_data2 = SearchLambdaOpt(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, GDparams, W, b, -3, -2);
% 

%% Bonus A: more exhaustive random searches

%lambda = [1e-2, 1e-3, 1e-4, 1e-5];
lambda = zeros(1,4);
n_s = zeros(1,4);

% Specifics for lambda calculations
l_min = -5;
dl = 3;

n_cycles = 4;

nb_tries = length(lambda)*length(n_s)*length(n_cycles); % Total nb of tries

max_accs = zeros(1, nb_tries); % Max accuracy from training

count = 1;

opt_data = zeros(nb_tries, 4); % Data collector

for i = 1:length(lambda)
    l = l_min + dl*rand(1,1);    
    lambda(i) = 10^l;
    for j = 1:length(n_s)
        n_s(j) = randi([500 1000]);
        for k = 1:length(n_cycles)
            disp(['--------- Try ', num2str(count), '/', num2str(nb_tries), ' ---------'])
            n_epochs = floor(2*n_cycles(k)*n_s(j)/(N/n_batch));
            GDparams = [n_batch, eta, n_epochs, n_s(j), eta_min, eta_max, gamma, p];
            [~, ~, ~, ~, ~, ~, ~, acc_valid] = train(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, GDparams, W, b, lambda(i));
            max_accs(count) = max(acc_valid);
            disp(['lambda = ', num2str(lambda(i)), ', n_s = ', num2str(n_s(j)), ..., 
                ', n_cycles = ', num2str(n_cycles(k)), ', acc_valid = ', num2str(max(acc_valid))])
            
            count = count + 1;
            
            %opt_data(count,:) = [lambda(i), n_s(j), n_cycles(k), max_accs(count)];
        end
    end
end

function PlotResults(J_train, J_valid, acc_train, acc_valid, L_train, L_valid, n_epochs)

t = 0:n_epochs;

figure(1)
plot(t, J_train, 'k', 'DisplayName', 'training');
hold on
plot(t, J_valid, 'r', 'DisplayName', 'validation');
xlabel('epoch');
ylabel('cost');
legend show
legend('Location', 'best')
grid on

figure(2)
plot(t, acc_train, 'k', 'DisplayName', 'training');
hold on
plot(t, acc_valid, 'r', 'DisplayName', 'validation');
xlabel('epoch');
ylabel('Accuracy');
legend show
legend('Location', 'best')
grid on

figure(3)
plot(t, L_train, 'k', 'DisplayName', 'training');
hold on
plot(t, L_valid, 'r', 'DisplayName', 'validation');
xlabel('epoch');
ylabel('loss');
legend show
legend('Location', 'best')
grid on
end

function lambda_data = SearchLambdaOpt(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, GDparams, W, b, l_min, l_max)
nb_tries = 15;
dl = l_max-l_min;

lambdas = zeros(nb_tries,1);
max_accs = zeros(nb_tries,1);

for i = 1:nb_tries
    disp(['Try ', num2str(i), '/', num2str(nb_tries)]);
    l = l_min + dl*rand(1,1);    
    lambdas(i) = 10^l;
    %[~, ~, ~, ~, ~, ~, ~, acc_valid] = train(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, GDparams, W, b, lambdas(i));    
    [~, ~, ~, ~, ~, ~, ~, acc_valid] = train(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, GDparams, W, b, lambdas(i));
    max_accs(i) = max(acc_valid);
    disp(['lambda = ', num2str(lambdas(i)), ', acc_valid = ', num2str(max_accs(i))]);
end

% Sorting data (for LaTeX tables)
lambda_data = [lambdas, max_accs];
[~, idx] = sort(lambda_data(:,2), 'descend');
lambda_data = round(lambda_data(idx,:), 4, 'significant');

end

function [X, Y, y] = LoadBatch(filename)
%Reads data from CIFAR-10 batch file
%Out: image and label data in separate files
%X: image pixel data, size d x N
%Y: One-hot representation of the label for each image, size K x N
%y: label for each image, size N x 1
%N = nb of images (10000), d = dimensionality of each image (32x32x3)
%K = nb of labels (10)

A = load(filename);
X = double(A.data')./255;
y = double(A.labels+1);
Y = eye(max(y));
Y = Y(y,:)';
end

function X = Normalize(X, mean_X, std_X)
    X = X - repmat(mean_X, [1, size(X, 2)]);
    X = X ./ repmat(std_X, [1, size(X, 2)]);
end

function [W, b] = InitializeParameters(m1, m2, sigma)
W = sigma*randn(m1, m2);
b = zeros(m1,1);
end

function [P, H, S] = FwdPass(X, W, b)
%Each column of X corresponds to an image, size d x n
%W, size K x d
%P contains proba for each label for the image in the corresponding X
%size K x n

S = cell(numel(b), 1);
H = cell(numel(b) - 1, 1);

S{1} = W{1}*X + b{1};
H{1} = max(0,S{1});

for i = 2:numel(S)
    S{i} = W{i}*H{i-1} + b{i};
    H{i} = max(0,S{i});
end

P = softmax(S{end});
end

function [P, U, H, S] = FwdPassInvDropout(X, W, b, p)
%Each column of X corresponds to an image, size d x n
%W, size K x d
%P contains proba for each label for the image in the corresponding X
%size K x n

S = cell(numel(b), 1);
H = cell(numel(b) - 1, 1); 
U = cell(numel(b) - 1, 1); 

S{1} = W{1}*X + b{1};
H{1} = max(0, S{1});
U{1} = (rand(size(H{1})) < p)/p;
H{1} = H{1}.*U{1};

for i = 2:numel(S)
    S{i} = W{i}*H{i-1} + b{i};
    H{i} = max(0, S{i});
    U{i} = (rand(size(H{i})) < p)/p;
    H{i} = H{i}.*U{i};
end

P = softmax(S{end});
end

function [L, J] = ComputeCost(P, Y, W, lambda)
% J scalar corresponding to the sum of the loss of the network's predictions
% for the images in X relative to the ground truth labels and regularization
% term on W
% P = FwdPass(X,W,b);

R = sumsqr(W);
L = -mean(sum(Y.*log(P)));
J = L + lambda*R;
end

function acc = ComputeAccuracy(P, y)
% P = FwdPass(X, W, b);
[~,k] = max(P);

acc = mean(k' == y);
end

function [grad_b, grad_W] = ComputeGradients(X, Y, W, b, lambda)
N = size(X,2);

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

for i = 1:numel(W)
    grad_W{i} = zeros(size(W{i}));
    grad_b{i} = zeros(size(b{i}));    
end

% [P, ~, H, ~] = FwdPassInvDropout(X, W, b, 0.5);
[P, H, ~] = FwdPass(X, W, b);

% Efficient computations for mini-batch gradient of loss, slide 30
G = -(Y-P);

for i = flip(2:numel(W))
    grad_W{i} = G*H{i-1}';
    grad_b{i} = G*ones(N,1);
    G = W{i}'*G;
    G = G.*(H{i-1} > 0);
end

grad_W{1} = G*X';
grad_b{1} = G*ones(N,1);

%Divide by nb of entries and add regularization term
for j = 1:length(W)
    grad_W{j} =  grad_W{j}/N + 2*lambda*W{j};
    grad_b{j} =  grad_b{j}/N;
end
end

function [b, W, GDparams, ustep] = MiniBatchGD(X_train, Y_train, GDparams, W, b, lambda, ustep)
n_batch = GDparams(1);
eta = GDparams(2);
n_s = GDparams(4);
eta_min = GDparams(5);
eta_max = GDparams(6);
%gamma = GDparams(7);
N = size(X_train,2);

%Momentum of W
% v_W = cell(numel(W), 1);
% v_W{1} = zeros(size(W{1}));
% v_W{2} = zeros(size(W{2}));

%Momentum of b
% v_b = cell(numel(b), 1);
% v_b{1} = zeros(size(b{1}));
% v_b{2} = zeros(size(b{2}));

for j = 1:N/n_batch
    eta = UpdateLearningRate(eta, ustep, n_s, eta_min, eta_max);
    j_start = (j-1)*n_batch + 1;
    j_end = j*n_batch;
    inds = j_start:j_end;
    Xbatch = X_train(:, inds);
    Ybatch = Y_train(:, inds);
    [grad_b, grad_W] = ComputeGradients(Xbatch, Ybatch, W, b, lambda);
    for k = 1:numel(W)
        W{k} = W{k} - eta*grad_W{k};
        b{k} = b{k} - eta*grad_b{k};
%         v_W{k} = gamma*v_W{k} + eta*grad_W{k};       
%         v_b{k} = gamma*v_b{k} + eta*grad_b{k};
%         W{k} = W{k} - v_W{k};
%         b{k} = b{k} - v_b{k};
    end
    ustep = ustep+1; %Update step
end

GDparams(2) = eta;
end

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

function new_eta = UpdateLearningRate(eta, t, n_s, eta_min, eta_max)
L = 0;
new_eta = eta;
deta = eta_max-eta_min;
while eta == new_eta
    if t >= (2*L*n_s) && t <= (n_s*(2*L+1))
        new_eta = eta_min + deta*(t-2*L*n_s)/n_s;
    elseif t >= (n_s*(2*L+1)) && t <= 2*(L+1)*n_s
        new_eta = eta_max - deta*(t-(2*L+1)*n_s)/n_s;
    end
    L = L + 1;
end
end

function [err_b, err_W] = ComputeRelativeError(ngrad_b, grad_b, ngrad_W, grad_W)
err_b = cell(numel(grad_b), 1);
err_W = cell(numel(grad_W), 1);

for i = numel(grad_b)
   err_b{i} =  norm(ngrad_b{i} - grad_b{i})/max(eps, norm(ngrad_b{i}) + norm(grad_b{i}));
   err_W{i} = norm(ngrad_W{i} - grad_W{i})/max(eps, norm(ngrad_W{i}) + norm(grad_W{i}));
end
end