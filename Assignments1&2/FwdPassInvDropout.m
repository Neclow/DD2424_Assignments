function [P, U, H, S] = FwdPassInvDropout(X, W, b, p)
%Each column of X corresponds to an image, size d x n
%W, size K x d
%P contains proba for each label for the image in the corresponding X
%size K x n

S = cell(numel(b), 1);
H = cell(numel(b) - 1, 1); 
U = cell(numel(b) - 2, 1);

S{1} = W{1}*X + b{1};
H{1} = max(0, S{1});
U{1} = (rand(size(H{1})) < p)/p;     % Inverted Dropout
S{1} = S{1}.*U{1};

for i = 2:numel(S)-1
    S{i} = W{i}*H{i-1} + b{i};
    H{i} = max(0, S{i});
    U{i} = (rand(size(H{i})) < p)/p; % Inverted Dropout
    H{i} = H{i}.*U{i};
end

S{end} = W{end}*H{end-1}+b{end};
H{end} = max(0, S{end});

P = softmax(S{end});
end