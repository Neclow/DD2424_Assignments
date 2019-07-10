%% 0.1 Read in the data
book_data = ExtractData('data/goblet_book.txt');
book_chars = unique(book_data);
GDparams.K = length(book_chars);

% Initialize map container to easily go btw character and one-hot encoding
keySet = num2cell(book_chars);
valueSet = int32(1:GDparams.K);
char_to_ind = containers.Map(keySet, valueSet);
ind_to_char = containers.Map(valueSet, keySet);

%% 0.2 Set hyper-parameters & initialize the RNN's parameters
rng(400)
GDparams.m = 100;                                   % Dimensionality of hidden state
GDparams.eta = .1;                                  % Learning rate
GDparams.epsilon = 1e-8;                            % Epsilon for AdaGrad
GDparams.seq_length = 25;                           % Length of input sequences
GDparams.syn_length = 1000;                         % Length of synthesized sequences
GDparams.sig = 0.01;                                % Std deviation
%GDparams.n_epochs = 2;                             % Nb of epochs
GDparams.n_steps = 3e5;                             % floor(length(book_data)*GDparams.n_epochs/GDparams.seq_length);
RNN.b = zeros(GDparams.m, 1);                       % Bias vector, mx1
RNN.c = zeros(GDparams.K, 1);                       % Bias vector, Kx1
RNN.U = randn(GDparams.m, GDparams.K)*GDparams.sig; % Weight matrix
RNN.W = randn(GDparams.m, GDparams.m)*GDparams.sig; % Weight matrix
RNN.V = randn(GDparams.K, GDparams.m)*GDparams.sig; % Weight matrix

%% 0.4 Implemented fwd & bwd pass of back-prop
% Debugging data
% X_chars = book_data(1:GDparams.seq_length);   
% Y_chars = book_data(2:GDparams.seq_length+1);
% X = OneHot(X_chars, char_to_ind);                     % Kxseq_length
% Y = OneHot(Y_chars, char_to_ind);                     % Kxseq_length
% h0 = zeros(GDparams.m, 1);
% 
% % Check gradients
% [a, h, p] = FwdPass(RNN, h0, x);
% grads = BwdPass(X,Y,RNN,a,h,p);
% num_grads = ComputeGradsNum(X, Y, RNN, 1e-4);
% errs = ComputeRelativeError(num_grads, grads);

%% 0.5 Train RNN using AdaGrad
disp('Starting training...')
[RNN, smooth_loss, Xs, hprevs] = train(book_data, RNN, GDparams, char_to_ind, ind_to_char);

disp('Training completed.')

[sl_min, t_min] = min(smooth_loss);
x0_min = Xs(:,t_min);
hprev_min = hprevs(:,t_min);
[~, gen_txt] = GenText(RNN, GDparams, hprev_min, x0_min, GDparams.syn_length, ind_to_char);
disp(gen_txt);

% disp('Plotting results ...')
% 
% t = 1:GDparams.n_steps;
% isNZ =(~smooth_loss==0);
% figure(1)
% plot(t(isNZ), smooth_loss(isNZ), 'k')
% xlabel('step')
% ylabel('smooth loss')
% %legend show
% %legend('Location', 'best')
% grid on
% 
% disp('Done.')

%% Functions for 0.1
function book_data = ExtractData(book_fname)
fid = fopen(book_fname, 'r');
book_data = fscanf(fid, '%c');
fclose(fid);
end

%% Functions for 0.3
function [a, h, p] = FwdPass(RNN, h0, X)
[N, tau] = size(X);

% Build a, h vectors
l = length(h0);
a = zeros(l, tau);
h = zeros(l, tau+1);
p = zeros(N, tau);
h(:,1) = h0;

for t = 1:tau
    xt = X(:, t);
    ht = h(:,t);
    [a(:,t), h(:,t+1), p(:,t)] = FwdPass2(RNN, ht, xt);
end
end

function [a, h, p] = FwdPass2(RNN, h0, x0)
a = RNN.W*h0 + RNN.U*x0 + RNN.b;
h = tanh(a);
o = RNN.V*h + RNN.c;
p = softmax(o);
end

function [Y, gen_txt] = GenText(RNN, GDparams, h0, x0, n, ind_to_char)
Y = zeros(GDparams.K, n);
gen_txt = '';
xnext = x0;
hnext = h0;
for t = 1:n
    [~, hnext, p] = FwdPass2(RNN, hnext, xnext);
    cp = cumsum(p);
    a = rand;
    ixs = find(cp-a > 0);
    ii = ixs(1);
    
    xnext = zeros(GDparams.K, 1);
    xnext(ii) = 1;
    Y(:,t) = xnext;
    gen_txt = [gen_txt, ind_to_char(ii)];
end
end

%% Functions for 0.4
function Y = OneHot(ys, char_to_ind)
Y = zeros(length(char_to_ind), length(ys));
inds = zeros(1, length(ys));
for i = 1:length(ys)
    inds(i) = char_to_ind(ys(i));
    Y(inds(i), i) = 1; 
end
end

function L = ComputeLoss(X, Y, RNN, h0)
tau = size(Y, 2);
L = 0;
h = h0;
for t = 1:tau
    xi = X(:,t);
    yi = Y(:,t);
    [~, h, pi] = FwdPass2(RNN, h, xi);
    L = L - log(yi'*pi);
end
end

function grads = BwdPass(X, Y, RNN, a, h, p)
% Grads for b, c, U, V, W
for f = fieldnames(RNN)'
    grads.(f{1}) = zeros(size(RNN.(f{1})));
end

tau = size(X, 2);

G = -(Y-p)';

% t = tau
htau = h(:,tau+1);   % h_{tau}
h2tau = h(:, tau);   % h_{tau-1}
atau = a(:,tau);
xtau = X(:,tau);
gtau = G(tau,:);     % grad_o

grads.c = gtau';
grads.V = gtau'*htau';

grad_h = gtau*RNN.V;
grad_a = grad_h*diag(1-tanh(atau).^2);

grads.b = grad_a';
grads.W = grad_a'*h2tau';
grads.U = grad_a'*xtau';

for i = flip(1:tau-1)
    gi = G(i,:);
    hi = h(:,i+1);
    h2i = h(:,i);
    ai = a(:,i);
    xi = X(:,i);
    
    grads.c = grads.c + gi';    
    grads.V = grads.V + gi'*hi';
    
    grad_h = gi*RNN.V + grad_a*RNN.W;
    grad_a = grad_h*diag(1-tanh(ai).^2);
    
    grads.b = grads.b + grad_a';
    grads.W = grads.W + grad_a'*h2i';
    grads.U = grads.U + grad_a'*xi';
end
end

function grads = ClipGradients(grads)
for f = fieldnames(grads)'
    grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
end
end

%% Functions for 0.5
function [RNN, smooth_loss, Xs, hprevs] = train(book_data, RNN, GDparams, char_to_ind, ind_to_char)
e = 1;                                              % Integer that keeps track of where in the book one is
max_e = length(book_data)-GDparams.seq_length-1;    % Maximum of e before an epoch is reached
m = InitializeSumSqr(RNN);                          % Initalize AdaGrad estimator
h0 = zeros(GDparams.m, 1);                          % Initial hidden state
smooth_loss = zeros(GDparams.n_steps, 1);           % Smooth loss

hprevs = zeros(GDparams.m, GDparams.n_steps);       % Collect hprevs for last task of Assignment
Xs = zeros(GDparams.K, GDparams.n_steps);           % Collect first characters for last task of Assignment

disp('Training... ')

epoch_count = 0;

for t = 1:GDparams.n_steps
    x = book_data(e:e+GDparams.seq_length-1); % Input characters
    y = book_data(e+1:e+GDparams.seq_length); % Labels for this input
    
    % One-hot encoding
    X = OneHot(x, char_to_ind);
    Y = OneHot(y, char_to_ind);
    
    % Set hprev
    if e == 1
        hprev = h0;
    else
        hprev = h(:, end);
    end
    
    % % Text generation
%     if mod(t-1, 1e4) == 0 
%         disp(['Step: ', num2str(t), '/', num2str(GDparams.n_steps), ... 
%             ', Smooth loss = ', num2str(smooth_loss(t))]);
%         [~, gen_txt] = GenText(RNN, GDparams, hprev, X(:,1), GDparams.syn_length, ind_to_char);
%         disp(gen_txt);
%     end
    
    % Data collection for last task
    Xs(:,t) = X(:,1);
    hprevs(:,t) = hprev;
    
    [a, h, p] = FwdPass(RNN, hprev, X);    % Fwd Pass
    grads = BwdPass(X, Y, RNN, a, h, p);   % Bwd Pass
    grads = ClipGradients(grads);          % Clip gradients
    
    % Update loss
    L = ComputeLoss(X, Y, RNN, hprev);
    if t == 1
        smooth_loss(t) = L;
    else
        smooth_loss(t) = 0.999*smooth_loss(t-1) + 0.001*L;
    end
    
    % Update RNN parameters with AdaGrad
    [RNN, m] = AdaGrad(RNN, m, grads, GDparams.eta, GDparams.epsilon);   
    
    % Update counter
    e = e + GDparams.seq_length;
    if e > max_e
        e = 1; % Restart looping through book
        epoch_count = epoch_count + 1;
        disp(['---------- Epochs of training completed: ', num2str(epoch_count) ' ---------'])
    end
    
    % Info on progess
    if mod(t-1, 1e4) == 0
        disp(['Step: ', num2str(t), '/', num2str(GDparams.n_steps), ... 
            ', Smooth loss = ', num2str(smooth_loss(t))]);
    end
end

% Text generation at end of training
disp('End of training... Last text generation:')
disp(['Step: ', num2str(GDparams.n_steps), '/', num2str(GDparams.n_steps), ... 
            ', Smooth loss = ', num2str(smooth_loss(end))]);
[~, gen_txt] = GenText(RNN, GDparams, hprev, X(:,1), GDparams.syn_length, ind_to_char);
disp(gen_txt);
end

function [RNN, m] = AdaGrad(RNN, m, grads, eta, epsilon)
for f = fieldnames(RNN)'
    m.(f{1}) = m.(f{1}) + grads.(f{1}).^2;
    RNN.(f{1}) = RNN.(f{1}) - eta*grads.(f{1})./sqrt(m.(f{1})+epsilon);
end
end

function m = InitializeSumSqr(RNN)
for f = fieldnames(RNN)'
    m.(f{1}) =  zeros(size(RNN.(f{1})));
end
end