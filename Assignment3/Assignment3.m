%% 0.1 Read in the data & get it ready
rng(400);
[all_names, ys] = ExtractNames;

C = unique(cell2mat(all_names));
d = numel(C);                            % dimensionality of 1-hot vector to encode each character
n_len = max(cellfun(@length,all_names)); % Max len of name in dataset
K = length(unique(ys));                  % Nb of classes to predict

% Initialize map container to easily go btw character and one-hot encoding
keySet = num2cell(C);
valueSet = int32(1:d);
char_to_ind = containers.Map(keySet, valueSet);

% One-hot-encoding of each character
Y = eye(max(ys));
Y = Y(ys,:)';

% Vectorized encoding of names
X = zeros(d*n_len, length(all_names));
for i = 1:length(all_names)
    X(:,i) = EncodeNames(all_names{i}, C, d, n_len);
end

% Partition data to create validation and training data
valid_inds = load('Validation_Inds.txt');
all_inds = 1:length(all_names);
common_inds = intersect(all_inds, valid_inds);
train_inds = setxor(all_inds, common_inds);
X_valid = PartitionData(X, valid_inds);
Y_valid = PartitionData(Y, valid_inds);
y_valid = PartitionData(ys', valid_inds);
y_valid = y_valid';
X_train = PartitionData(X, train_inds);
Y_train = PartitionData(Y, train_inds);
y_train = PartitionData(ys', train_inds);
y_train = y_train';

%% 0.2 Set hyper-parameters & initialize ConvNet's parameters
n1 = 20;            % Nb of filters applied at layer 1                  
n2 = 20;            % Nb of filters applied at layer 2
k1 = 3;             % Width of filters applied at layer 1
k2 = 3;             % Width of filters applied at layer 2

n_len1 = n_len - k1 + 1;
n_len2 = n_len1 - k2 + 1;
n_lens = [n_len, n_len1, n_len2];
fsize = n2*n_len2; % Number of elements in X(2)

sig1 = sqrt(2/k1);      % He initialization
sig2 = sqrt(2/n2);      % He initialization
sig3 = sqrt(2/fsize);   % He initialization

ConvNet.F{1} = randn(d, k1, n1)*sig1;
ConvNet.F{2} = randn(n1, k2, n2)*sig2;
ConvNet.W = randn(K, fsize)*sig3;

n_batch = 100;
eta = 0.001;         % Learning rate
%n_epochs = 120;
gamma = 0.9;         % momentum term
n_steps = 100;
%n_steps = n_epochs*floor(size(X_train,2)/n_batch);
comp = true;        % training w/ compensation or not

%% 0.4 Implement the forward & backward pass of back-prop
% X_batch = X_train(:,1:n_batch); Y_batch = Y_train(:,1:n_batch);
% h = 1e-6;
% MFs = {MakeMFMatrix(ConvNet.F{1}, n_len), MakeMFMatrix(ConvNet.F{2}, n_len1)};
% [d1, k1, n1] = size(ConvNet.F{1});
% MX1 = PreComputeMX(X_batch, d1, k1, n1, n_len);
% [grad_F, grad_W] = ComputeGradients(Y_batch, X_batch, ConvNet, MFs, MX1, n_lens);
% grad_F1 = grad_F{1}(:); grad_F2 = grad_F{2}(:);
% Gs = NumericalGradient(X_batch, Y_batch, ConvNet, h, n_len, n_len1);
% ngrad_F1 = Gs{1}(:); ngrad_F2 = Gs{2}(:); ngrad_W = Gs{3};
% err_F1 = ComputeRelativeError(ngrad_F1, grad_F1); %1.3798e-8
% err_F2 = ComputeRelativeError(ngrad_F2, grad_F2); %6.8765e-9
% err_W = ComputeRelativeError(ngrad_W, grad_W);    %6.1327e-10

%% 0.5 Train using mini-batch gradient descent with momentum
if comp
    n_min = GetSampleSize(X_train, Y_train);
    n_epochs = floor(n_steps/(K*n_min/n_batch));
    n_steps = n_epochs*floor(K*n_min/n_batch);
    GDparams = [n_batch, eta, n_epochs, gamma, n_steps];
    disp('Training mode: balanced.')
    [ConvNet, L_train, L_valid, acc_train, acc_valid] = trainBalanced(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, ConvNet, GDparams, n_lens);
else
    disp('Training mode: unbalanced.')
    n_epochs = floor(n_steps/(size(X_train,2)/n_batch));
    n_steps = n_epochs*floor(size(X_train,2)/n_batch);
    GDparams = [n_batch, eta, n_epochs, gamma, n_steps];
    [ConvNet, L_train, L_valid, acc_train, acc_valid] = trainImbalanced(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, ConvNet, GDparams, n_lens);
end

disp('Training completed.')

disp('Plotting results ...')

t = 1:n_steps+1;
isNZ =(~L_train==0);

figure(1)
plot(t(isNZ), L_train(isNZ), 'k', 'DisplayName', 'training')
hold on
plot(t(isNZ), L_valid(isNZ), 'r', 'DisplayName', 'validation')
xlabel('step')
ylabel('loss')
legend show
legend('Location', 'best')
grid on

figure(2)
[maxy,idx] = max(acc_valid);
plot(t(isNZ), acc_train(isNZ), 'k', 'DisplayName', 'training')
hold on
plot(t(isNZ), acc_valid(isNZ), 'r', 'DisplayName', 'validation')
%text(idx, maxy, ['max. = ', num2str(maxy)]);
xlabel('step')
ylabel('Accuracy')
legend show
legend('Location', 'best')
grid on

disp('Done.')

MFs = {MakeMFMatrix(ConvNet.F{1}, n_len), MakeMFMatrix(ConvNet.F{2}, n_len1)};

% Compute validation accuracy for each class
accs = ComputeClassAccuracy(X_valid, Y_valid, y_valid, MFs, ConvNet);

% Confusion matrix
C = ComputeConfMat(X_valid, y_valid, MFs, ConvNet);

%% Functions for 0.1
function [all_names, ys] = ExtractNames
%Reads contents of ascii_names.txt
%Puts all names in cell array all_names
%Puts corresponding labels in ys

data_fname = 'ascii_names.txt';

fid = fopen(data_fname,'r');
S = fscanf(fid,'%c');
fclose(fid);
names = strsplit(S, '\n');
if length(names{end}) < 1        
    names(end) = [];
end
ys = zeros(length(names), 1);
all_names = cell(1, length(names));
for i=1:length(names)
    nn = strsplit(names{i}, ' ');
    l = str2num(nn{end});
    if length(nn) > 2
        name = strjoin(nn(1:end-1));
    else
        name = nn{1};
    end
    name = lower(name);
    ys(i) = l;
    all_names{i} = name;
end

disp('Saving the data...')
tic
save('assignment3_names.mat', 'ys', 'all_names');
toc
end

function vecName = EncodeNames(name, C, d, n_len)
%Y: one-hot-encoding of each character
matName = zeros(d, n_len);

%If test name has length greater than n_len, pick first n_len chars
if length(name) > n_len
    name = name(1:n_len);
end

for i = 1:length(name)
    [~, b] = find(C == name(i));
    matName(b,i) = 1;
end

vecName = matName(:);
end

function Xp = PartitionData(X, inds)

Xp = zeros(size(X,1), length(inds));

for i = 1:length(inds)
    Xp(:,i) = X(:,inds(i));
end
end

%% 0.3 Construct convolution matrices
function MF = MakeMFMatrix(F, n_len)
% Size of MF: (n_len-k+1)*nf x n_len*dd
[dd, k, nf] = size(F);

rows = (n_len-k+1)*nf;
cols = n_len*dd;
MF = zeros(rows, cols);

VF = [];
for i = 1:nf
    subF = F(:,:,i);
    f = subF(:)';
    VF = [VF; f];
end

for j = 0:n_len-k
    MF(j*nf+1:(j+1)*nf,1+dd*j:size(VF,2)+dd*j) = VF;
end
end

function MX = MakeMXMatrix(x_input, d, k, nf, n_len)
X_input = reshape(x_input, [d, n_len]);
rows = (n_len-k+1)*nf;
cols = k*nf*d;

MX = zeros(rows, cols);

for i = 0:n_len-k
    subMX = zeros(nf,cols);
    for j = 1:nf
         subX = X_input(:, i+1:i+k);        
         subx = subX(:)';
         subMX(j, (j-1)*length(subx)+1:j*length(subx)) = subx;
    end
    MX(i*nf+1:(i+1)*nf,:) = subMX;
end
end

function MX = MakeMXMatrix2(x_input, d, k, n_len)
X_input = reshape(x_input, [d, n_len]);
rows = (n_len-k+1);
cols = k*d;


MX = zeros(rows, cols);
for i = 1:n_len-k+1
    subX = X_input(:, i:i+k-1);
    subx = subX(:)';
    MX(i, :) = subx;
end
end

%% Functions for 0.4
function [X_batch1, X_batch2, P_batch] = FwdPass(X_batch, MFs, W)
% X_batch: (n_len*d) x n -> all vectorised input data
% Y_batch: one-hot encoding of labels of each ex
% W: 2D weight matrix for last fully connected layer

X_batch1 = max(MFs{1}*X_batch, 0);
X_batch2 = max(MFs{2}*X_batch1,0);

S_batch = W*X_batch2;

P_batch = softmax(S_batch);
end

function L = ComputeLoss(Ys_batch, P_batch)
L = -mean(sum(Ys_batch.*log(P_batch)));
end

function acc = ComputeAccuracy(P, y)
[~,k] = max(P);

acc = mean(k' == y);
end

function [grad_F, grad_W] = ComputeGradients(Y_batch, X_batch, ConvNet, MFs, MX1, n_lens)
n_len1 = n_lens(2);
n_len2 = n_lens(3);

[X_batch1, X_batch2, P_batch] = FwdPass(X_batch, MFs, ConvNet.W);

grad_F = {zeros(1, length(ConvNet.F{1}(:))), zeros(1, length(ConvNet.F{2}(:)))};

[n1, k2, n2] = size(ConvNet.F{2});

G_batch = -(Y_batch-P_batch);

N = size(X_batch, 2);

grad_W = (1/N)*G_batch*X_batch2';

G_batch = ConvNet.W'*G_batch;
G_batch = G_batch.*(X_batch2 > 0);

for j = 1:N
    gj = G_batch(:,j);
    Gj = reshape(gj, [n2 n_len2]).';
    xj = X_batch1(:,j);
    V = sparse(MakeMXMatrix2(xj, n1, k2, n_len1))'*Gj;
    grad_F{2} = grad_F{2} + V(:)'/N;
end

G_batch = MFs{2}'*G_batch;
G_batch = G_batch.*(X_batch1 > 0);

for j = 1:N
    gj = G_batch(:,j);
    v = gj'*MX1{1,j};
    grad_F{1} = grad_F{1} + v/N;
end

grad_F{1} = reshape(grad_F{1}, size(ConvNet.F{1}));
grad_F{2} = reshape(grad_F{2}, size(ConvNet.F{2}));
end

function Gs = NumericalGradient(X_inputs, Ys, ConvNet, h, n_len, n_len1)

try_ConvNet = ConvNet;
Gs = cell(length(ConvNet.F)+1, 1);

for l=1:length(ConvNet.F)
    try_convNet.F{l} = ConvNet.F{l};
    
    Gs{l} = zeros(size(ConvNet.F{l}));
    nf = size(ConvNet.F{l},  3);
    
    for i = 1:nf        
        try_ConvNet.F{l} = ConvNet.F{l};
        F_try = squeeze(ConvNet.F{l}(:, :, i));
        G = zeros(numel(F_try), 1);
        
        for j=1:numel(F_try)
            F_try1 = F_try;
            F_try1(j) = F_try(j) - h;
            try_ConvNet.F{l}(:, :, i) = F_try1; 
            
            MFs = {MakeMFMatrix(try_ConvNet.F{1}, n_len), MakeMFMatrix(try_ConvNet.F{2}, n_len1)}; 
            [~, ~, P_batch] = FwdPass(X_inputs, MFs, try_ConvNet.W);
            l1 = ComputeLoss(Ys, P_batch);
            
            F_try2 = F_try;
            F_try2(j) = F_try(j) + h;            
            
            try_ConvNet.F{l}(:, :, i) = F_try2;
            MFs = {MakeMFMatrix(try_ConvNet.F{1}, n_len), MakeMFMatrix(try_ConvNet.F{2}, n_len1)}; 
            [~, ~, P_batch] = FwdPass(X_inputs, MFs, try_ConvNet.W);
            l2 = ComputeLoss(Ys, P_batch); 
            
            G(j) = (l2 - l1) / (2*h);
            try_ConvNet.F{l}(:, :, i) = F_try;
        end
        Gs{l}(:, :, i) = reshape(G, size(F_try));
    end
end

%% compute the gradient for the fully connected layer
W_try = ConvNet.W;
G = zeros(numel(W_try), 1);
for j=1:numel(W_try)
    W_try1 = W_try;
    W_try1(j) = W_try(j) - h;
    try_ConvNet.W = W_try1; 
    
    [~, ~, P_batch] = FwdPass(X_inputs, MFs, try_ConvNet.W);
    l1 = ComputeLoss(Ys, P_batch);
            
    W_try2 = W_try;
    W_try2(j) = W_try(j) + h;            
            
    try_ConvNet.W = W_try2;
    [~, ~, P_batch] = FwdPass(X_inputs, MFs, try_ConvNet.W);
    l2 = ComputeLoss(Ys, P_batch);           
            
    G(j) = (l2 - l1) / (2*h);
    try_ConvNet.W = W_try;
end
Gs{end} = reshape(G, size(W_try));
end

function err = ComputeRelativeError(ngrad, agrad)
err = norm(ngrad - agrad)/max(eps, norm(ngrad) + norm(agrad));
end

%% Functions for 0.5
function MX1s = PreComputeMX(X, d1, k1, n1, n_len)
disp('Pre-computing MX...');
tic
N = size(X, 2);
MX1s = cell(1,N);
for j = 1:N
    xj = X(:,j); 
    MX1s{1,j} = sparse(MakeMXMatrix(xj, d1, k1, n1, n_len));
end
toc
end

function [ConvNet, MFs, ustep, L_train, L_valid, acc_train, acc_valid] = MiniBatchGD(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, ConvNet, MFs, MX1s, GDparams, n_lens, ustep, L_train, L_valid, acc_train, acc_valid)
n_batch = GDparams(1);
eta = GDparams(2);
gamma = GDparams(4);                                          % Momentum coef
n_steps = GDparams(5);
N = size(X_train,2);
n_len = n_lens(1);
n_len1 = n_lens(2);

v_W = zeros(size(ConvNet.W));                                 % Momentum of W
v_F = {zeros(size(ConvNet.F{1})), zeros(size(ConvNet.F{2}))}; % Momentum of F

for j = 1:N/n_batch
    tic
    j_start = (j-1)*n_batch + 1;
    j_end = j*n_batch;
    inds = j_start:j_end;
    Xbatch = X_train(:, inds);
    Ybatch = Y_train(:, inds);     
    MX1 = MX1s(:, inds);
    
    [grad_F, grad_W] = ComputeGradients(Ybatch, Xbatch, ConvNet, MFs, MX1, n_lens);
    
    v_W = gamma*v_W + eta*grad_W;
    ConvNet.W = ConvNet.W - v_W;
    for k = 1:numel(ConvNet.F)    
        v_F{k} = gamma*v_F{k} + eta*grad_F{k};
        ConvNet.F{k} = ConvNet.F{k} - v_F{k};
    end
    
    % Update filter matrix
    MFs = {MakeMFMatrix(ConvNet.F{1}, n_len), MakeMFMatrix(ConvNet.F{2}, n_len1)};
    
    ustep = ustep+1; % Update step
    
    toc
    if mod(ustep, 500) == 0
        [L_train, L_valid, acc_train, acc_valid] = CollectData(X_train, Y_train, y_train, L_train, acc_train, X_valid, Y_valid, y_valid, L_valid, acc_valid, MFs, ConvNet, ustep, n_steps);
    end
end
end

function [ConvNet, L_train, L_valid, acc_train, acc_valid] = trainImbalanced(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, ConvNet, GDparams, n_lens)
n_epochs = GDparams(3);
n_steps = GDparams(5);

ustep = 1; %Update step

n_len = n_lens(1);
n_len1 = n_lens(2);

% Loss, Accuracy data
L_train = zeros(n_steps, 1);
L_valid = zeros(n_steps, 1);
acc_train = zeros(n_steps, 1);
acc_valid = zeros(n_steps, 1);

% Initial filter matrix
MFs = {MakeMFMatrix(ConvNet.F{1}, n_len), MakeMFMatrix(ConvNet.F{2}, n_len1)};

% Pre-compute MX for training
[d1, k1, n1] = size(ConvNet.F{1});
MX1s = PreComputeMX(X_train, d1, k1, n1, n_len);

% Initial loss/cost/accuracy values (before training)
[L_train, L_valid, acc_train, acc_valid] = CollectData(X_train, Y_train, y_train, L_train, acc_train, X_valid, Y_valid, y_valid, L_valid, acc_valid, MFs, ConvNet, ustep, n_steps);

disp('Training... ')

for t = 1:n_epochs
    [ConvNet, MFs, ustep, L_train, L_valid, acc_train, acc_valid] = MiniBatchGD(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, ConvNet, MFs, MX1s, GDparams, n_lens, ustep, L_train, L_valid, acc_train, acc_valid);
end

% Computing losses and accuracies at end of training
disp('End of training...');
[L_train, L_valid, acc_train, acc_valid] = CollectData(X_train, Y_train, y_train, L_train, acc_train, X_valid, Y_valid, y_valid, L_valid, acc_valid, MFs, ConvNet, ustep, n_steps);
end

function [ConvNet, L_train, L_valid, acc_train, acc_valid] = trainBalanced(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, ConvNet, GDparams, n_lens)
n_epochs = GDparams(3);
n_steps = GDparams(5);

ustep = 1; %Update step

n_len = n_lens(1); 
n_len1 = n_lens(2);

% Loss, Accuracy data
L_train = zeros(n_steps, 1);
L_valid = zeros(n_steps, 1);
acc_train = zeros(n_steps, 1);
acc_valid = zeros(n_steps, 1);

% Initial filter matrix
MFs = {MakeMFMatrix(ConvNet.F{1}, n_len), MakeMFMatrix(ConvNet.F{2}, n_len1)};

% Pre-compute MX for training
[d1, k1, n1] = size(ConvNet.F{1});
MX1s = PreComputeMX(X_train, d1, k1, n1, n_len);

% Initial loss/cost/accuracy values (before training)
[L_train, L_valid, acc_train, acc_valid] = CollectData(X_train, Y_train, y_train, L_train, acc_train, X_valid, Y_valid, y_valid, L_valid, acc_valid, MFs, ConvNet, ustep, n_steps);

disp('Training... ')

for t = 1:n_epochs
    [X_s, Y_s, y_s] = SampleTrainingSet(X_train, Y_train, y_train);
    [ConvNet, MFs, ustep, L_train, L_valid, acc_train, acc_valid] = MiniBatchGD(X_s, Y_s, y_s, X_valid, Y_valid, y_valid, ConvNet, MFs, MX1s, GDparams, n_lens, ustep, L_train, L_valid, acc_train, acc_valid);
end

% Computing losses and accuracies at end of training
disp('End of training...');
[L_train, L_valid, acc_train, acc_valid] = CollectData(X_train, Y_train, y_train, L_train, acc_train, X_valid, Y_valid, y_valid, L_valid, acc_valid, MFs, ConvNet, ustep, n_steps);
end

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

%% Functions for 0.6
% function p = CompensationCoefs(X_train, Y_train)
% K = size(Y_train, 1);
% n = zeros(1, K);
% p = zeros(1, K);
% for i = 1:size(X_train,2)
%     [~, y] = max(Y_train(:,i));
%     n(y) = n(y) + 1;
% end
% for y = 1:K
%     p(y) = 1/(n(y) * K);
% end
% end

function n_min = GetSampleSize(X_train, Y_train)
K = size(Y_train, 1);
n = zeros(1, K);

for i = 1:size(X_train,2)
    [~, y] = max(Y_train(:,i));
    n(y) = n(y) + 1;
end
n_min = min(n);
end

function [X_s, Y_s, y_s] = SampleTrainingSet(X_train, Y_train, y_train)
K = size(Y_train, 1);
n = zeros(1, K);

for i = 1:size(X_train,2)
    [~, y] = max(Y_train(:,i));
    n(y) = n(y) + 1;
end
n_min = min(n);

inds = zeros(1, K*length(n));

inds(1:n_min) = randsample(1:n(1), n_min);

n_start = n(1);

for i = 1:length(n)-1
    n_end = n_start + n(i+1);
    inds(1+n_min*i:n_min*(i+1)) = randsample(n_start+1:n_end, n_min);
    n_start = n_end;
end

X_s = X_train(:,inds);
Y_s = Y_train(:,inds);
y_s = y_train(inds,:);
end

function accs = ComputeClassAccuracy(X_valid, Y_valid, y_valid, MFs, ConvNet)
%Validation: 14 examples per class
n = 14;
K = size(Y_valid, 1);

accs = zeros(1, K);
[~, ~, P] = FwdPass(X_valid, MFs, ConvNet.W);

n_start = 0;
for i = 0:K-1
    n_end = n_start + n;
    accs(i+1) = ComputeAccuracy(P(:,1+n_start:n_end), y_valid(1+n_start:n_end,:));
    n_start = n_end;
end
end

function C = ComputeConfMat(X_valid, y_valid, MFs, ConvNet)
[~, ~, P_valid] = FwdPass(X_valid, MFs, ConvNet.W);
[~, k] = max(P_valid);

C = confusionmat(k, y_valid);
end